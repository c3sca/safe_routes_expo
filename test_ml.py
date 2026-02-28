"""
test_ml.py – Standalone tests for the three ML components in SafeRoutes.

No database, no OSMnx download needed.
Run with:  python test_ml.py

Tests:
  1. Gradient Boosting (SafetyScorePredictor)
  2. Online Learning   (OnlineSafetyLearner / River)
  3. Dijkstra routing  (find_route on a synthetic graph)
"""
import sys
import math
import random
import traceback
import numpy as np
import pandas as pd
import networkx as nx

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"  [{status} ] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# 1. GRADIENT BOOSTING
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_data(n_streets: int = 30, ratings_per_tod: int = 5):
    """
    Build realistic synthetic DataFrames that match what ml_pipeline expects.
    Streets have varied crime/lighting profiles; ratings are generated from them.
    """
    random.seed(0)
    np.random.seed(0)

    streets = []
    for i in range(n_streets):
        crime   = round(random.uniform(0.05, 0.80), 3)
        light   = round(random.uniform(0.30, 0.95), 3)
        # Ground-truth composite: roughly 0.5*(1-crime) + 0.3*(1-crime) + 0.2*light
        composite = float(np.clip(0.5 * (1 - crime) + 0.3 * (1 - crime) + 0.2 * light, 0, 1))
        streets.append({
            "street_name":            f"Street_{i}",
            "latitude":               55.93 + random.uniform(0, 0.04),
            "longitude":              -3.23 + random.uniform(0, 0.09),
            "crime_rate_normalised":  crime,
            "lighting_score":         light,
            "avg_user_score":         round(1 - crime * 0.6, 3),
            "composite_safety_score": composite,
        })

    streets_df = pd.DataFrame(streets)

    ratings = []
    times = ["day", "evening", "night"]
    for s in streets:
        for tod in times:
            night_penalty = {"day": 0.0, "evening": 0.05, "night": 0.15}[tod]
            mean_raw = max(1.2, min(4.8, 5 - 3 * s["crime_rate_normalised"] - night_penalty))
            for uid in range(ratings_per_tod):
                score = int(round(np.clip(np.random.normal(mean_raw, 0.5), 1, 5)))
                ratings.append({
                    "user_id":      uid + 1,
                    "street_name":  s["street_name"],
                    "safety_score": score,
                    "time_of_day":  tod,
                    "latitude":     s["latitude"] + random.uniform(-0.001, 0.001),
                    "longitude":    s["longitude"] + random.uniform(-0.001, 0.001),
                })

    ratings_df = pd.DataFrame(ratings)
    return ratings_df, streets_df


def test_gradient_boosting():
    section("1. GRADIENT BOOSTING  (SafetyScorePredictor)")

    from models.ml_pipeline import SafetyScorePredictor

    ratings_df, streets_df = make_synthetic_data(n_streets=40, ratings_per_tod=6)

    predictor = SafetyScorePredictor()

    # ── Train ─────────────────────────────────────────────────────────────────
    metrics = predictor.train(ratings_df, streets_df)

    check("Model trained (trained flag set)", predictor.trained)
    check("RMSE returned",  metrics["rmse"] is not None,
          f"RMSE = {metrics['rmse']:.4f}" if metrics["rmse"] else "")
    check("R² returned",    metrics["r2"]   is not None,
          f"R²   = {metrics['r2']:.4f}" if metrics["r2"] else "")
    check("RMSE < 0.25",    metrics["rmse"] is not None and metrics["rmse"] < 0.25,
          f"{metrics['rmse']:.4f}")
    check("R² > 0.3",       metrics["r2"]   is not None and metrics["r2"]   > 0.3,
          f"{metrics['r2']:.4f}")

    # ── Predict: scores in valid range ────────────────────────────────────────
    coords = [
        (55.9523, -3.1920, "day",     "Princes St (day)"),
        (55.9473, -3.1895, "night",   "Cowgate (night)"),
        (55.9535, -3.2005, "evening", "George St (evening)"),
    ]
    print()
    for lat, lng, tod, label in coords:
        score = predictor.predict_score(lat, lng, tod)
        ok = 0.0 <= score <= 1.0
        check(f"Score in [0,1] for {label}", ok, f"score = {score:.3f}")

    # ── Safer street should score higher ──────────────────────────────────────
    safe_score   = predictor.predict_score(55.9355, -3.1879, "night",
                                           crime_rate=0.15, lighting=0.68)
    unsafe_score = predictor.predict_score(55.9473, -3.1895, "night",
                                           crime_rate=0.75, lighting=0.50)
    check("Safe street scores higher than unsafe (night)",
          safe_score > unsafe_score,
          f"safe={safe_score:.3f}  unsafe={unsafe_score:.3f}")

    # ── Untrained fallback ────────────────────────────────────────────────────
    fresh = SafetyScorePredictor()
    fb = fresh.predict_score(55.95, -3.19, "night", crime_rate=0.3, lighting=0.7)
    check("Untrained fallback returns heuristic in [0,1]",
          0.0 <= fb <= 1.0, f"fallback = {fb:.3f}")

    # ── Feature importance available ──────────────────────────────────────────
    importances = predictor.model.feature_importances_
    check("Feature importances non-empty", len(importances) == 7)
    top_feat_idx = int(np.argmax(importances))
    feat_names = ["latitude", "longitude", "time_encoded",
                  "crime_rate", "lighting", "num_ratings", "avg_rating"]
    print(f"\n  Top feature: {feat_names[top_feat_idx]}  "
          f"(importance = {importances[top_feat_idx]:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ONLINE LEARNING (River)
# ─────────────────────────────────────────────────────────────────────────────

def test_online_learning():
    section("2. ONLINE LEARNING  (River – Hoeffding Adaptive Tree)")

    try:
        from models.online_learner import OnlineSafetyLearner
    except ImportError as e:
        print(f"  [SKIP] Could not import OnlineSafetyLearner: {e}")
        return

    learner = OnlineSafetyLearner()

    if learner.model is None:
        print("  [SKIP] River not installed – online learning disabled.")
        return

    check("Learner initialised", learner.n_seen == 0)

    # ── No prediction before warm-up ──────────────────────────────────────────
    pred_cold = learner.predict_one(55.95, -3.19, "night", 0.5, 0.5, 0.5)
    check("Returns None before 10 observations", pred_cold is None)

    # ── Feed safe-street observations ────────────────────────────────────────
    random.seed(1)
    safe_scores  = [0.75, 0.80, 0.70, 0.78, 0.72, 0.76, 0.82, 0.69, 0.74, 0.77,
                    0.80, 0.75]
    unsafe_scores = [0.20, 0.25, 0.18, 0.22, 0.15, 0.23, 0.19, 0.21, 0.24, 0.17,
                     0.20, 0.22]

    # Warm up with safe street
    for s in safe_scores:
        learner.learn_one(55.9355, -3.1879, "day", 0.15, 0.68, 0.75, s)

    check(f"n_seen after {len(safe_scores)} obs", learner.n_seen == len(safe_scores),
          f"n_seen = {learner.n_seen}")

    # Predictions now available
    safe_pred = learner.predict_one(55.9355, -3.1879, "day", 0.15, 0.68, 0.75)
    check("Prediction available after >=10 obs", safe_pred is not None,
          f"pred = {safe_pred:.3f}" if safe_pred is not None else "None")

    # Feed unsafe street
    for s in unsafe_scores:
        learner.learn_one(55.9473, -3.1895, "night", 0.75, 0.50, 0.25, s)

    unsafe_pred = learner.predict_one(55.9473, -3.1895, "night", 0.75, 0.50, 0.25)
    check("Prediction for unsafe street available", unsafe_pred is not None,
          f"pred = {unsafe_pred:.3f}" if unsafe_pred is not None else "None")

    if safe_pred is not None and unsafe_pred is not None:
        check("Safe street pred > unsafe street pred",
              safe_pred > unsafe_pred,
              f"safe={safe_pred:.3f}  unsafe={unsafe_pred:.3f}")

    # ── Running MAE is tracked ─────────────────────────────────────────────────
    stats = learner.get_stats()
    check("get_stats() has n_seen key",  "n_seen"    in stats)
    check("get_stats() has mae key",     "mae"       in stats)
    check("get_stats() has available key", "available" in stats)
    print(f"\n  Stats: {stats}")

    # ── Predictions bounded [0, 1] ────────────────────────────────────────────
    for _ in range(20):
        lat  = random.uniform(55.93, 55.97)
        lng  = random.uniform(-3.23, -3.14)
        tod  = random.choice(["day", "evening", "night"])
        pred = learner.predict_one(lat, lng, tod,
                                   random.uniform(0, 1), random.uniform(0, 1), 0.5)
        if pred is not None:
            if not (0.0 <= pred <= 1.0):
                check("Prediction clamped to [0,1]", False, f"pred={pred}")
                break
    else:
        check("All random predictions clamped to [0,1]", True)

    # ── warm_up on DataFrames ──────────────────────────────────────────────────
    ratings_df, streets_df = make_synthetic_data(n_streets=10, ratings_per_tod=3)
    learner2 = OnlineSafetyLearner()
    n_before = learner2.n_seen
    learner2.warm_up(ratings_df, streets_df)
    check("warm_up increases n_seen", learner2.n_seen > n_before,
          f"n_seen = {learner2.n_seen}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. DIJKSTRA ROUTING (synthetic graph – no OSMnx download needed)
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_graph() -> nx.MultiDiGraph:
    """
    Build a small synthetic Edinburgh-like walking graph for testing.

    Layout (node IDs = integers):

        1 ── 2 ── 3
        |         |
        4 ── 5 ── 6

    Nodes carry x (lng) and y (lat) attributes like an OSMnx graph.
    Edges carry 'length' (metres) and 'safety_score'.
    We make the top path (1→2→3) unsafe and the bottom path (1→4→5→6→3)
    safe so we can verify the routing algorithm picks the safer one when
    alpha > 0.
    """
    G = nx.MultiDiGraph()

    # lat/lng for each node
    nodes = {
        1: (55.950, -3.210),
        2: (55.950, -3.200),   # top middle – unsafe
        3: (55.950, -3.190),
        4: (55.940, -3.210),
        5: (55.940, -3.200),
        6: (55.940, -3.190),
    }
    for nid, (lat, lng) in nodes.items():
        G.add_node(nid, y=lat, x=lng)

    def haversine(n1, n2):
        lat1, lng1 = nodes[n1]
        lat2, lng2 = nodes[n2]
        R = 6_371_000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
             + math.cos(phi1) * math.cos(phi2)
             * math.sin(math.radians(lng2 - lng1) / 2) ** 2)
        return 2 * R * math.asin(math.sqrt(a))

    # Top row: short but unsafe
    top_edges = [(1, 2), (2, 3)]
    # Bottom row: longer but safe
    bot_edges = [(1, 4), (4, 5), (5, 6), (6, 3)]

    for u, v in top_edges:
        dist = haversine(u, v)
        G.add_edge(u, v, key=0, length=dist, safety_score=0.20)   # dangerous
        G.add_edge(v, u, key=0, length=dist, safety_score=0.20)

    for u, v in bot_edges:
        dist = haversine(u, v)
        G.add_edge(u, v, key=0, length=dist, safety_score=0.85)   # safe
        G.add_edge(v, u, key=0, length=dist, safety_score=0.85)

    return G


def test_dijkstra():
    section("3. DIJKSTRA ROUTING  (synthetic graph)")

    from models.route_engine import compute_edge_weight, colour_route_segments

    G = _make_synthetic_graph()

    # ── compute_edge_weight ───────────────────────────────────────────────────
    safe_edge   = {"length": 100.0, "safety_score": 0.9}
    unsafe_edge = {"length": 100.0, "safety_score": 0.1}

    safe_cost   = compute_edge_weight(safe_edge,   "length", alpha=0.7)
    unsafe_cost = compute_edge_weight(unsafe_edge, "length", alpha=0.7)
    check("Safe edge costs less than unsafe (alpha=0.7)",
          safe_cost < unsafe_cost,
          f"safe={safe_cost:.1f}  unsafe={unsafe_cost:.1f}")

    # With alpha=0 (pure shortest path), costs should be equal (just distance)
    cost_a0_safe   = compute_edge_weight(safe_edge,   "length", alpha=0.0)
    cost_a0_unsafe = compute_edge_weight(unsafe_edge, "length", alpha=0.0)
    check("alpha=0 -> pure distance (costs equal for same length)",
          abs(cost_a0_safe - cost_a0_unsafe) < 1e-6,
          f"safe={cost_a0_safe:.1f}  unsafe={cost_a0_unsafe:.1f}")

    # ── Dijkstra on synthetic graph ───────────────────────────────────────────
    # Nodes 1 and 3 are diagonally opposite corners
    # Top path (1→2→3): short, dangerous (safety=0.20)
    # Bottom path (1→4→5→6→3): longer, safe (safety=0.85)

    # We call find_route, which uses ox.nearest_nodes internally – that won't
    # work on a plain nx graph.  Instead, call nx.dijkstra_path directly with
    # our weight function (same logic as find_route) and verify the result.

    from models.route_engine import compute_edge_weight

    def weight_fn_safe(*args):
        data = args[2]
        if isinstance(data, dict) and 0 in data:
            return compute_edge_weight(data[0], "length", alpha=0.9)
        return compute_edge_weight(data, "length", alpha=0.9)

    def weight_fn_short(*args):
        data = args[2]
        if isinstance(data, dict) and 0 in data:
            return compute_edge_weight(data[0], "length", alpha=0.0)
        return compute_edge_weight(data, "length", alpha=0.0)

    path_safe  = nx.dijkstra_path(G, 1, 3, weight=weight_fn_safe)
    path_short = nx.dijkstra_path(G, 1, 3, weight=weight_fn_short)

    # Safe routing should prefer the bottom (longer) path
    check("Safe routing avoids dangerous top path",
          2 not in path_safe,
          f"path = {path_safe}")

    # Shortest routing should use the top (shorter) path
    check("Shortest routing uses direct top path",
          2 in path_short,
          f"path = {path_short}")

    check("Safe path has more nodes (longer route)",
          len(path_safe) > len(path_short),
          f"safe_len={len(path_safe)}  short_len={len(path_short)}")

    # ── colour_route_segments ─────────────────────────────────────────────────
    safe_path_nodes  = [1, 4, 5, 6, 3]   # safety_score=0.85 → green
    unsafe_path_nodes = [1, 2, 3]         # safety_score=0.20 → red

    segs_safe   = colour_route_segments(G, safe_path_nodes)
    segs_unsafe = colour_route_segments(G, unsafe_path_nodes)

    check("colour_route_segments returns correct segment count for safe path",
          len(segs_safe) == 4, f"len={len(segs_safe)}")
    check("colour_route_segments returns correct segment count for unsafe path",
          len(segs_unsafe) == 2, f"len={len(segs_unsafe)}")

    all_green = all(s["color"] == "#22c55e" for s in segs_safe)
    all_red   = all(s["color"] == "#ef4444" for s in segs_unsafe)
    check("Safe segments are green  (safety=0.85 >= 0.7)", all_green,
          f"colors = {[s['color'] for s in segs_safe]}")
    check("Unsafe segments are red  (safety=0.20 < 0.4)",  all_red,
          f"colors = {[s['color'] for s in segs_unsafe]}")

    # ── No-path case ──────────────────────────────────────────────────────────
    # Build an isolated node that cannot be reached
    G_broken = G.copy()
    G_broken.add_node(99, y=55.99, x=-3.50)   # disconnected node
    try:
        nx.dijkstra_path(G_broken, 1, 99)
        check("NetworkXNoPath raised for disconnected node", False)
    except nx.NetworkXNoPath:
        check("NetworkXNoPath raised for disconnected node", True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\nSafeRoutes ML Test Suite")
    print("No database / OSMnx download required.\n")

    results = []

    for name, fn in [
        ("Gradient Boosting", test_gradient_boosting),
        ("Online Learning",   test_online_learning),
        ("Dijkstra routing",  test_dijkstra),
    ]:
        try:
            fn()
            results.append((name, True))
        except Exception:
            print(f"\n  [EXCEPTION in {name}]")
            traceback.print_exc()
            results.append((name, False))

    section("SUMMARY")
    all_ok = True
    for name, ok in results:
        status = PASS if ok else FAIL
        print(f"  [{status} ] {name}")
        if not ok:
            all_ok = False

    print()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
