"""
online_learner.py – Real-time incremental learning with River.

River is an online machine learning library: models update incrementally
from a single observation at a time, without retraining on the full dataset.

Why this matters for SafeRoutes:
  When a user submits a new safety rating the app immediately updates the
  River model. No batch retraining needed. The model continuously improves.

Model: River's Hoeffding Adaptive Tree Regressor (HATR)
  - A decision-tree variant designed for streaming data
  - Uses ADWIN drift detection to adapt when the data distribution changes
    (e.g. an area that was safe becomes less safe over time)

Alternative: River's SGDRegressor (stochastic gradient descent) or
             a Pipeline with StandardScaler + LinearRegression for speed.
  We use a small ensemble (BaggingRegressor over HoeffdingAdaptiveTreeRegressor)
  for slightly better accuracy on sparse streaming data.

Features fed to the model:
  - latitude, longitude          (geographic position)
  - time_day, time_eve, time_ngt (one-hot for time_of_day)
  - crime_rate                   (from area record)
  - lighting                     (from area record)
  - hist_avg_rating              (running average of past ratings)
"""
import os
import pickle
from typing import Optional

try:
    from river import preprocessing, linear_model, metrics, ensemble, tree
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    print("[River] River not installed – online learning will be disabled.")


def _make_features(lat: float, lng: float, time_of_day: str,
                   crime_rate: float, lighting: float,
                   hist_avg: float) -> dict:
    """
    Build a River-compatible feature dict for a single observation.

    River uses plain Python dicts as feature vectors (not numpy arrays).
    One-hot encode time_of_day so the linear model can handle categories.
    """
    return {
        "latitude":  lat,
        "longitude": lng,
        # One-hot encode time of day
        "time_day":  1.0 if time_of_day == "day"     else 0.0,
        "time_eve":  1.0 if time_of_day == "evening" else 0.0,
        "time_ngt":  1.0 if time_of_day == "night"   else 0.0,
        "crime_rate": crime_rate,
        "lighting":   lighting,
        "hist_avg":   hist_avg,
    }


class OnlineSafetyLearner:
    """
    Wraps a River streaming regression pipeline.

    Pipeline:
      StandardScaler (online, adapts to running mean/variance)
        ↓
      BaggingRegressor (ensemble of HoeffdingAdaptiveTreeRegressors)

    The Hoeffding Adaptive Tree handles concept drift (e.g. an area that
    gets safer over time as development improves lighting) via ADWIN.
    """

    def __init__(self):
        if not RIVER_AVAILABLE:
            self.model   = None
            self.mae     = None
            self.n_seen  = 0
            return

        # Build the River pipeline
        # StandardScaler adapts its mean and variance incrementally
        self.model = (
            preprocessing.StandardScaler()
            | ensemble.BaggingRegressor(
                model=tree.HoeffdingAdaptiveTreeRegressor(
                    grace_period=50,
                    delta=1e-5,
                    leaf_prediction="adaptive",
                ),
                n_models=5,
                seed=42,
            )
        )

        # Online MAE metric to track model accuracy over time
        self.mae    = metrics.MAE()
        self.n_seen = 0    # number of observations the model has learned from

    def learn_one(self, lat: float, lng: float, time_of_day: str,
                  crime_rate: float, lighting: float,
                  hist_avg: float, true_score: float) -> Optional[float]:
        """
        Update the model with a single new observation.

        Parameters
        ----------
        true_score : float
            The normalised safety score (0–1) from the new user rating.

        Returns
        -------
        float or None : prediction *before* this update (used for metric tracking)
        """
        if self.model is None:
            return None

        x = _make_features(lat, lng, time_of_day, crime_rate, lighting, hist_avg)
        y = true_score

        # Predict first (evaluate before learning)
        y_pred = self.model.predict_one(x)

        # Then learn from this observation (online update)
        self.model.learn_one(x, y)
        self.n_seen += 1

        # Update running MAE
        if y_pred is not None:
            self.mae.update(y, y_pred)

        return y_pred

    def predict_one(self, lat: float, lng: float, time_of_day: str,
                    crime_rate: float, lighting: float,
                    hist_avg: float) -> Optional[float]:
        """
        Predict the safety score (0–1) for a location without updating the model.

        Returns None if the model hasn't seen enough data yet.
        """
        if self.model is None or self.n_seen < 10:
            return None

        x = _make_features(lat, lng, time_of_day, crime_rate, lighting, hist_avg)
        pred = self.model.predict_one(x)
        if pred is None:
            return None
        return float(min(1.0, max(0.0, pred)))

    def warm_up(self, ratings_df, streets_df) -> None:
        """
        Pre-train the River model on historical seed data so it starts
        with reasonable weights rather than zero knowledge.

        Parameters
        ----------
        ratings_df : pd.DataFrame  [latitude, longitude, time_of_day,
                                    street_name, safety_score]
        streets_df : pd.DataFrame  [street_name, crime_rate_normalised,
                                    lighting_score, avg_user_score]
        """
        if self.model is None or ratings_df.empty:
            return

        print(f"[River] Warming up on {len(ratings_df)} historical ratings...")

        # Build a lookup for street features
        street_lookup = streets_df.set_index("street_name").to_dict("index")

        for _, row in ratings_df.iterrows():
            street_info = street_lookup.get(row.get("street_name", ""), {})
            crime     = float(street_info.get("crime_rate_normalised", 0.5))
            lighting  = float(street_info.get("lighting_score", 0.5))
            hist_avg  = float(street_info.get("avg_user_score", 0.5))
            norm_score = (row["safety_score"] - 1) / 4.0   # 1–5 → 0–1

            self.learn_one(
                lat=row["latitude"],
                lng=row["longitude"],
                time_of_day=row["time_of_day"],
                crime_rate=crime,
                lighting=lighting,
                hist_avg=hist_avg,
                true_score=norm_score,
            )

        print(f"[River] Warm-up complete. n_seen={self.n_seen}  MAE={self.mae.get():.4f}")

    def save(self, path: str) -> None:
        """Serialise the River model to disk using pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[River] Model saved to {path}")

    @staticmethod
    def load(path: str) -> "OnlineSafetyLearner":
        """Load a previously saved River model."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_stats(self) -> dict:
        """Return a summary dict of current model state."""
        return {
            "n_seen": self.n_seen,
            "mae":    round(self.mae.get(), 4) if (self.mae and self.n_seen > 0) else None,
            "available": RIVER_AVAILABLE,
        }


def initialise_online_learner(app, db_instance, river_model_path: str) -> OnlineSafetyLearner:
    """
    Load an existing River model from disk, or create and warm up a new one.

    Called at app startup.
    """
    import pandas as pd
    from models.database import SafetyRating, Street

    # Try loading an existing model
    if os.path.exists(river_model_path):
        print(f"[River] Loading existing model from {river_model_path}")
        try:
            learner = OnlineSafetyLearner.load(river_model_path)
            print(f"[River] Loaded. n_seen={learner.n_seen}")
            return learner
        except Exception as e:
            print(f"[River] Failed to load: {e}. Creating fresh model.")

    # Create a new model and warm it up on historical data
    learner = OnlineSafetyLearner()

    with app.app_context():
        ratings_rows = SafetyRating.query.all()
        streets_rows = Street.query.all()

        ratings_df = pd.DataFrame([{
            "latitude":    r.latitude,
            "longitude":   r.longitude,
            "time_of_day": r.time_of_day,
            "street_name": r.street_name,
            "safety_score": r.safety_score,
        } for r in ratings_rows])

        streets_df = pd.DataFrame([{
            "street_name":           s.name,
            "crime_rate_normalised": s.crime_rate_normalised,
            "lighting_score":        s.lighting_score,
            "avg_user_score":        s.avg_user_score,
        } for s in streets_rows])

        learner.warm_up(ratings_df, streets_df)

    learner.save(river_model_path)
    return learner
