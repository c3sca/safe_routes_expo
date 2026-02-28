"""
app.py – SafeRoutes Flask application entry point.

Startup sequence:
  1. Create/seed the SQLite database if empty
  2. Run the full ML pipeline (ALS, Bayesian, sklearn)
  3. Initialise the River online learner
  4. Load (or download) the Edinburgh street graph from OSMnx
  5. Annotate the graph with safety scores
  6. Start the Flask server

Routes:
  GET  /                   → map / routing page
  GET  /rate               → area rating page
  GET  /heatmap            → safety heatmap page
  GET  /dashboard          → user dashboard
  GET  /login              → login page
  GET  /register           → register page
  POST /api/route          → compute a safe route
  POST /api/rate           → submit a safety rating
  GET  /api/areas          → list all areas with scores
  GET  /api/heatmap_data   → heatmap point data
  GET  /api/area_info      → click-on-map area info
"""
import json
import os
from datetime import datetime

from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, flash)
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, current_user)

from config import Config
from models.database import db, User, SafetyRating, Street, Route

# ── App factory ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)

# Initialise extensions
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access that page."

# Global model references (populated at startup)
_ml_models   = {}   # {'als', 'bayes', 'predictor', 'metrics'}
_online_learner = None
_graph          = None


@login_manager.user_loader
def load_user(user_id: str) -> User:
    return User.query.get(int(user_id))


# ── Startup ───────────────────────────────────────────────────────────────────

def startup():
    """Run all initialisation tasks before the first request."""
    global _ml_models, _online_learner, _graph

    with app.app_context():
        # 1. Create tables
        db.create_all()

        # 2. Seed database if empty
        from seed_data import seed
        seed(app, db)

        # 3. ML pipeline (ALS + Bayesian + sklearn)
        from models.ml_pipeline import run_full_pipeline
        _ml_models = run_full_pipeline(app, db, Config)

        # 4. River online learner
        from models.online_learner import initialise_online_learner
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        _online_learner = initialise_online_learner(app, db, Config.RIVER_MODEL_PATH)

        # 5. Edinburgh street graph
        from models.route_engine import load_edinburgh_graph, annotate_graph_safety
        _graph = load_edinburgh_graph(Config.GRAPH_PATH, Config.OSM_BBOX)

        # 6. Annotate graph edges with safety scores
        predictor = _ml_models.get("predictor")
        if _graph is not None and predictor is not None:
            from models.route_engine import annotate_graph_safety
            annotate_graph_safety(_graph, predictor, time_of_day="night")

    print("\n[app] SafeRoutes is ready.\n")


# ── Page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main map and routing page."""
    areas = Street.query.all()
    return render_template("index.html", areas=areas)


@app.route("/rate")
@login_required
def rate():
    """Area rating submission page."""
    return render_template("rate.html")


@app.route("/heatmap")
def heatmap():
    """Safety heatmap page."""
    return render_template("heatmap.html")


@app.route("/dashboard")
@login_required
def dashboard():
    """User profile and history dashboard."""
    user_ratings = (SafetyRating.query
                    .filter_by(user_id=current_user.id)
                    .order_by(SafetyRating.created_at.desc())
                    .limit(20)
                    .all())
    user_routes = (Route.query
                   .filter_by(user_id=current_user.id)
                   .order_by(Route.created_at.desc())
                   .limit(10)
                   .all())
    return render_template("dashboard.html",
                           ratings=user_ratings,
                           routes=user_routes)


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user, remember=True)
            next_page = request.args.get("next")
            return redirect(next_page or url_for("index"))
        flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username   = request.form.get("username", "").strip()
        email      = request.form.get("email", "").strip()
        password   = request.form.get("password", "")
        university = request.form.get("university", "University of Edinburgh").strip()

        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")

        if User.query.filter_by(username=username).first():
            flash("That username is already taken.", "error")
            return render_template("register.html")

        if User.query.filter_by(email=email).first():
            flash("An account with that email already exists.", "error")
            return render_template("register.html")

        user = User(username=username, email=email, university=university)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        flash("Welcome to SafeRoutes!", "success")
        return redirect(url_for("index"))

    return render_template("register.html")


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/route", methods=["POST"])
def api_route():
    """
    Compute a safety-optimised walking route.

    Request JSON:
      { "start_lat": float, "start_lng": float,
        "end_lat":   float, "end_lng":   float,
        "alpha":     float (0–1),
        "algorithm": "astar" | "dijkstra" }

    Response JSON:
      { "coordinates": [[lat,lng], ...],
        "distance_km":  float,
        "safety_score": float,
        "segments":     [...],   (colour-coded segments)
        "algorithm":    str }
    """
    data = request.get_json(force=True)
    try:
        start_lat = float(data["start_lat"])
        start_lng = float(data["start_lng"])
        end_lat   = float(data["end_lat"])
        end_lng   = float(data["end_lng"])
        alpha     = float(data.get("alpha",     Config.DEFAULT_ALPHA))
        algorithm = data.get("algorithm", "astar")
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid request: {e}"}), 400

    if _graph is None:
        return jsonify({"error": "Street graph not available. Please try again later."}), 503

    from models.route_engine import find_route, colour_route_segments
    result = find_route(
        _graph,
        start_lat, start_lng,
        end_lat,   end_lng,
        alpha=alpha
    )

    if result is None:
        return jsonify({"error": "No route found between those points."}), 404

    # Colour-code the segments
    segments = colour_route_segments(_graph, result["nodes"])
    result["segments"] = segments

    # Save route for logged-in users
    if current_user.is_authenticated:
        route_record = Route(
            user_id=current_user.id,
            start_lat=start_lat, start_lng=start_lng,
            end_lat=end_lat,     end_lng=end_lng,
            route_geometry=json.dumps(result["coordinates"]),
            safety_score=result["safety_score"],
            distance_km=result["distance_km"],
            algorithm=algorithm,
        )
        db.session.add(route_record)
        db.session.commit()

    return jsonify(result)


@app.route("/api/rate", methods=["POST"])
@login_required
def api_rate():
    """
    Submit a new safety rating for a location.

    Triggers:
      • River online learner update
      • Bayesian posterior update for the nearest area
      • Area composite score recalculation

    Request JSON:
      { "latitude":    float,
        "longitude":   float,
        "safety_score": int (1–5),
        "time_of_day":  "day"|"evening"|"night",
        "comment":      str (optional) }
    """
    data = request.get_json(force=True)
    try:
        lat   = float(data["latitude"])
        lng   = float(data["longitude"])
        score = int(data["safety_score"])
        tod   = data.get("time_of_day", "day")
        comment = data.get("comment", "")
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid request: {e}"}), 400

    if not 1 <= score <= 5:
        return jsonify({"error": "safety_score must be 1–5"}), 400

    # Find the nearest street
    streets = Street.query.all()
    from models.safety_scorer import find_nearest_street
    nearest     = find_nearest_street(lat, lng, streets)
    street_name = nearest.name if nearest else "Unknown"

    # Persist the rating
    rating = SafetyRating(
        user_id=current_user.id,
        latitude=lat,
        longitude=lng,
        street_name=street_name,
        safety_score=score,
        time_of_day=tod,
        comment=comment or None,
    )
    db.session.add(rating)
    db.session.commit()

    # ── River online update ────────────────────────────────────────────
    if _online_learner is not None and nearest is not None:
        norm_score = (score - 1) / 4.0
        _online_learner.learn_one(
            lat=lat, lng=lng,
            time_of_day=tod,
            crime_rate=nearest.crime_rate_normalised,
            lighting=nearest.lighting_score,
            hist_avg=nearest.avg_user_score,
            true_score=norm_score,
        )
        try:
            _online_learner.save(Config.RIVER_MODEL_PATH)
        except Exception:
            pass   # non-fatal

    # ── Bayesian street score update ───────────────────────────────────
    if nearest is not None:
        bayes = _ml_models.get("bayes")
        from models.safety_scorer import update_street_score
        new_composite = update_street_score(nearest, db.session, bayes,
                                            Config.SCORE_W1,
                                            Config.SCORE_W2,
                                            Config.SCORE_W3)
    else:
        new_composite = None

    return jsonify({
        "success":          True,
        "street_name":      street_name,
        "new_composite_score": new_composite,
        "river_stats": _online_learner.get_stats() if _online_learner else {},
    })


@app.route("/api/areas")
def api_areas():
    """Return all street records with their safety scores."""
    streets = Street.query.all()
    return jsonify([{
        "name":             s.name,
        "lat":              s.latitude,
        "lng":              s.longitude,
        "composite_score":  round(s.composite_safety_score, 3),
        "avg_user_score":   round(s.avg_user_score, 3),
        "crime_rate":       round(s.crime_rate_normalised, 3),
        "lighting":         round(s.lighting_score, 3),
    } for s in streets])


@app.route("/api/heatmap_data")
def api_heatmap_data():
    """Return heatmap points for Leaflet.heat."""
    mode    = request.args.get("mode", "composite")
    streets = Street.query.all()
    from models.safety_scorer import get_heatmap_data
    points = get_heatmap_data(streets, filter_mode=mode)
    return jsonify(points)


@app.route("/api/area_info")
def api_area_info():
    """
    Return info about the nearest area to a clicked map point.

    Query params: lat=float&lng=float
    """
    try:
        lat = float(request.args["lat"])
        lng = float(request.args["lng"])
    except (KeyError, ValueError):
        return jsonify({"error": "lat and lng are required"}), 400

    streets = Street.query.all()
    from models.safety_scorer import find_nearest_street
    nearest = find_nearest_street(lat, lng, streets)
    if nearest is None:
        return jsonify({"error": "No streets found"}), 404

    # Recent ratings for this street
    recent_ratings = (SafetyRating.query
                      .filter_by(street_name=nearest.name)
                      .order_by(SafetyRating.created_at.desc())
                      .limit(5)
                      .all())

    return jsonify({
        "name":            nearest.name,
        "composite_score": round(nearest.composite_safety_score, 3),
        "avg_user_score":  round(nearest.avg_user_score, 3),
        "crime_rate":      round(nearest.crime_rate_normalised, 3),
        "lighting":        round(nearest.lighting_score, 3),
        "recent_ratings": [{
            "score":       r.safety_score,
            "time_of_day": r.time_of_day,
            "comment":     r.comment or "",
        } for r in recent_ratings],
    })


@app.route("/api/predict_score")
def api_predict_score():
    """
    Predict safety score for any point using the sklearn model.

    Query params: lat, lng, time_of_day
    """
    try:
        lat = float(request.args.get("lat", 55.95))
        lng = float(request.args.get("lng", -3.19))
        tod = request.args.get("time_of_day", "night")
    except ValueError:
        return jsonify({"error": "Invalid coordinates"}), 400

    predictor = _ml_models.get("predictor")
    if predictor is None:
        return jsonify({"score": 0.5, "source": "default"})

    # Look up street features for context
    streets = Street.query.all()
    from models.safety_scorer import find_nearest_street
    nearest = find_nearest_street(lat, lng, streets)
    crime   = nearest.crime_rate_normalised if nearest else 0.5
    light   = nearest.lighting_score        if nearest else 0.5
    avg     = nearest.avg_user_score        if nearest else 0.5

    score = predictor.predict_score(lat, lng, tod, crime, light, avg_rating=avg)
    return jsonify({"score": round(score, 3), "source": "sklearn"})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    startup()
    app.run(debug=False, host="127.0.0.1", port=5000)
