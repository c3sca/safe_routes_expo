"""
ml_pipeline.py – Core Machine Learning pipeline for SafeRoutes.

Contains three main components:
  1. ALS collaborative filtering  – calibrate user rating biases per street
  2. Bayesian score estimation    – robust posterior safety scores per street
  3. Scikit-learn Gradient Boosting – predict safety for any map point

Mathematical notes are included inline so the code is self-explaining.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────────────────────
# 1.  ALS COLLABORATIVE FILTERING
# ──────────────────────────────────────────────────────────────

class ALSCollaborativeFilter:
    """
    Alternating Least Squares (ALS) matrix factorisation.

    The idea:
      We have a sparse rating matrix  R  (users × streets).
      R[u, s] is the safety score user u gave to street s (or missing).

      ALS factorises R ≈ U · Vᵀ, where:
        U  ∈ ℝ^{n_users   × k}  – user latent factors
        V  ∈ ℝ^{n_streets × k}  – street latent factors
        k  = n_factors

      We alternate between:
        • Fixing V, solving for each row u of U analytically (ridge regression):
              u_i = (Vᵀ Cᵢ V + λI)⁻¹ Vᵀ Cᵢ r_i
          where Cᵢ is a diagonal confidence matrix (1 where rated, 0 elsewhere)
          and r_i is user i's ratings vector.
        • Fixing U, solving for each row v of V analogously.

      λ (lambda) is L2 regularisation to prevent overfitting.

    After training, predicted rating for (user u, street s) = U[u] · V[s].
    """

    def __init__(self, n_factors: int = 10, n_iter: int = 20, reg: float = 0.01):
        self.n_factors    = n_factors
        self.n_iter       = n_iter
        self.reg          = reg           # λ regularisation
        self.U            = None          # user factor matrix
        self.V            = None          # street factor matrix
        self.user_index   = {}            # user_id    → row index in R
        self.street_index = {}            # street_name → col index in R
        self.street_names = []

    def _build_matrix(self, ratings_df: pd.DataFrame) -> np.ndarray:
        """
        Convert a ratings DataFrame into a dense user×street matrix.

        ratings_df columns: user_id, street_name, safety_score (1–5)
        Missing entries are 0 (treated as unobserved, not as "zero safety").
        """
        users   = ratings_df["user_id"].unique().tolist()
        streets = ratings_df["street_name"].unique().tolist()
        self.user_index   = {uid: i for i, uid in enumerate(users)}
        self.street_index = {s: j   for j, s   in enumerate(streets)}
        self.street_names = streets

        n_users   = len(users)
        n_streets = len(streets)
        R = np.zeros((n_users, n_streets))

        for _, row in ratings_df.iterrows():
            u = self.user_index[row["user_id"]]
            s = self.street_index[row["street_name"]]
            R[u, s] = (row["safety_score"] - 1) / 4.0   # normalise 1–5 → 0–1

        return R

    def _solve_row(self, fixed: np.ndarray, R_slice: np.ndarray,
                   mask: np.ndarray) -> np.ndarray:
        """
        Solve for a single row of the factor matrix analytically.

        For user i (or street j), with the other factor matrix Fixed (V or U):
          min_x  Σ_{observed j} (R[i,j] - x · Fixed[j])²  +  λ ||x||²

        Closed-form solution (ordinary ridge regression):
          x = (Fᵀ C F + λ I)⁻¹ Fᵀ C r

        where:
          F = Fixed[mask]        – rows of Fixed where ratings exist
          C = identity           – uniform confidence for all observed ratings
          r = R_slice[mask]      – observed ratings for this row/column
        """
        F = fixed[mask]
        r = R_slice[mask]

        if F.shape[0] == 0:
            return np.zeros(self.n_factors)

        k = self.n_factors
        A = F.T @ F + self.reg * np.eye(k)
        b = F.T @ r
        return np.linalg.solve(A, b)

    def fit(self, ratings_df: pd.DataFrame) -> None:
        """Train ALS on a DataFrame of (user_id, street_name, safety_score)."""
        R = self._build_matrix(ratings_df)
        n_users, n_streets = R.shape
        k = self.n_factors

        rng = np.random.default_rng(42)
        self.U = rng.normal(0, 0.1, (n_users, k))
        self.V = rng.normal(0, 0.1, (n_streets, k))

        obs = R > 0
        errors = []
        for iteration in range(self.n_iter):
            for u in range(n_users):
                self.U[u] = self._solve_row(self.V, R[u, :], obs[u, :])
            for s in range(n_streets):
                self.V[s] = self._solve_row(self.U, R[:, s], obs[:, s])

            R_pred = self.U @ self.V.T
            err = np.sqrt(np.mean((R[obs] - R_pred[obs]) ** 2))
            errors.append(err)

        print(f"[ALS] Final reconstruction RMSE (observed): {errors[-1]:.4f}")

    def predict(self, user_id: int, street_name: str) -> float:
        """
        Predict the normalised safety score (0–1) for a (user, street) pair.
        Falls back to the street's average latent score if user is unknown.
        """
        s_idx = self.street_index.get(street_name)
        if s_idx is None:
            return 0.5   # completely unknown street

        if user_id in self.user_index:
            u_idx = self.user_index[user_id]
            score = float(self.U[u_idx] @ self.V[s_idx])
        else:
            score = float(np.mean(self.U, axis=0) @ self.V[s_idx])

        return float(np.clip(score, 0.0, 1.0))

    def get_street_scores(self) -> dict:
        """
        Return ALS-derived latent safety score for every street.
        Score = mean predicted rating across all users.
        Returns dict: street_name → score (0–1).
        """
        R_pred = self.U @ self.V.T          # (n_users, n_streets)
        street_scores = np.clip(np.mean(R_pred, axis=0), 0, 1)
        return {name: float(street_scores[j])
                for name, j in self.street_index.items()}


# ──────────────────────────────────────────────────────────────
# 2.  BAYESIAN SAFETY SCORE ESTIMATOR
# ──────────────────────────────────────────────────────────────

class BayesianSafetyEstimator:
    """
    Bayesian estimation of street safety using a Beta–Bernoulli conjugate model.

    Model:
      • Prior:   safety_prob ~ Beta(α₀, β₀)
        α₀ and β₀ encode our prior belief about the city-wide average.
        If city average is μ and we trust it with N₀ pseudo-observations:
            α₀ = μ * N₀,  β₀ = (1 - μ) * N₀

      • Likelihood: each rating (after normalising to [0,1]) is treated as a
        Bernoulli-like observation. To make it compatible with the Beta-Bernoulli
        model we binarise: r_normalised ≥ 0.5 → "success" (safe), else "failure".

      • Posterior (closed form – Beta is conjugate to Bernoulli):
            α_post = α₀ + Σ successes
            β_post = β₀ + Σ failures
            E[safety] = α_post / (α_post + β_post)

    This approach:
      • Regularises towards the city mean when data is sparse
      • Converges to the empirical mean as more ratings accumulate
      • Provides a credible interval if needed
    """

    def __init__(self, city_avg_safety: float = 0.6, prior_strength: float = 5.0):
        self.city_avg  = city_avg_safety
        self.strength  = prior_strength
        self.alpha0 = city_avg_safety * prior_strength
        self.beta0  = (1.0 - city_avg_safety) * prior_strength

    def estimate(self, normalised_scores: list) -> float:
        """
        Compute the posterior expected safety probability for one street.

        Parameters
        ----------
        normalised_scores : list of float
            User ratings for this street, already normalised to [0, 1].

        Returns
        -------
        float : posterior mean safety score in [0, 1]
        """
        successes = sum(1 for s in normalised_scores if s >= 0.5)
        failures  = len(normalised_scores) - successes

        alpha_post = self.alpha0 + successes
        beta_post  = self.beta0  + failures

        return alpha_post / (alpha_post + beta_post)

    def estimate_all_streets(self, ratings_df: pd.DataFrame) -> dict:
        """
        Compute Bayesian posterior safety scores for all streets.

        Parameters
        ----------
        ratings_df : DataFrame with columns [street_name, safety_score]
            safety_score is on the 1–5 scale.

        Returns
        -------
        dict : street_name → bayesian_score (0–1)
        """
        results = {}
        for street_name, group in ratings_df.groupby("street_name"):
            normalised = [(s - 1) / 4.0 for s in group["safety_score"]]
            results[street_name] = self.estimate(normalised)
        return results


# ──────────────────────────────────────────────────────────────
# 3.  SCIKIT-LEARN GRADIENT BOOSTING SAFETY PREDICTOR
# ──────────────────────────────────────────────────────────────

class SafetyScorePredictor:
    """
    Gradient Boosting Regressor trained on street seed data.

    Features:
      latitude, longitude, time_of_day_encoded,
      crime_rate_normalised, lighting_score, num_ratings, avg_rating

    Target:
      composite_safety_score (0–1)

    Once trained, this model can predict a safety score for any arbitrary
    (lat, lng, time_of_day) – which is critical for scoring every edge of the
    Edinburgh road network during routing.
    """

    def __init__(self):
        self.model   = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )
        self.le_time = LabelEncoder()
        self.trained = False

    def _build_features(self, ratings_df: pd.DataFrame,
                        streets_df: pd.DataFrame) -> tuple:
        """
        Join ratings to street data and build the feature matrix X and target y.

        Returns (X: np.ndarray, y: np.ndarray)
        """
        agg = (
            ratings_df
            .groupby(["street_name", "time_of_day"])
            .agg(num_ratings=("safety_score", "count"),
                 avg_rating=("safety_score", "mean"))
            .reset_index()
        )
        agg["avg_rating_norm"] = (agg["avg_rating"] - 1) / 4.0

        merged = agg.merge(streets_df, on="street_name", how="left")

        all_times = ["day", "evening", "night"]
        self.le_time.fit(all_times)
        merged["time_encoded"] = self.le_time.transform(merged["time_of_day"])

        feature_cols = [
            "latitude", "longitude", "time_encoded",
            "crime_rate_normalised", "lighting_score",
            "num_ratings", "avg_rating_norm",
        ]
        X = merged[feature_cols].values
        y = merged["composite_safety_score"].values
        return X, y

    def train(self, ratings_df: pd.DataFrame, streets_df: pd.DataFrame) -> dict:
        """
        Train the Gradient Boosting model on seed data.

        Parameters
        ----------
        ratings_df : DataFrame  [user_id, street_name, safety_score, time_of_day]
        streets_df : DataFrame  [street_name, latitude, longitude,
                                 crime_rate_normalised, lighting_score,
                                 composite_safety_score]

        Returns
        -------
        dict with 'rmse' and 'r2' on the held-out test set.
        """
        X, y = self._build_features(ratings_df, streets_df)

        if len(X) < 5:
            print("[sklearn] Not enough data to train. Skipping.")
            return {"rmse": None, "r2": None}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        self.trained = True

        y_pred = self.model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = float(r2_score(y_test, y_pred))

        print(f"[sklearn] GradientBoosting  RMSE={rmse:.4f}  R²={r2:.4f}")
        return {"rmse": rmse, "r2": r2}

    def predict_score(self, lat: float, lng: float,
                      time_of_day: str = "night",
                      crime_rate: float = 0.5,
                      lighting: float   = 0.5,
                      num_ratings: int  = 10,
                      avg_rating: float = 0.5) -> float:
        """
        Predict the composite safety score (0–1) for any point on the map.
        Falls back to a simple heuristic if the model isn't trained yet.
        """
        if not self.trained:
            return float(np.clip(0.5 * (1 - crime_rate) + 0.5 * lighting, 0, 1))

        time_enc = self.le_time.transform([time_of_day])[0]
        X = np.array([[lat, lng, time_enc, crime_rate, lighting,
                        num_ratings, avg_rating]])
        score = float(self.model.predict(X)[0])
        return float(np.clip(score, 0.0, 1.0))

    def save(self, path: str) -> None:
        """Pickle the trained model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[sklearn] Model saved to {path}")

    @staticmethod
    def load(path: str) -> "SafetyScorePredictor":
        """Load a pickled model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ──────────────────────────────────────────────────────────────
# 4.  PIPELINE ORCHESTRATOR
# ──────────────────────────────────────────────────────────────

def run_full_pipeline(app, db_instance, config) -> dict:
    """
    Orchestrate the full ML pipeline at startup:
      1. Load data from the database
      2. Run ALS collaborative filtering
      3. Run Bayesian score estimation
      4. Compute composite safety scores and store back to DB
      5. Train the sklearn Gradient Boosting model
      6. Save the sklearn model to disk

    Returns a dict containing the trained model objects.
    """
    from models.database import SafetyRating, Street

    with app.app_context():
        # ── Load ratings from DB ──────────────────────────────────────────────
        ratings_rows = SafetyRating.query.all()
        ratings_df = pd.DataFrame([{
            "user_id":      r.user_id,
            "street_name":  r.street_name,
            "safety_score": r.safety_score,
            "time_of_day":  r.time_of_day,
            "latitude":     r.latitude,
            "longitude":    r.longitude,
        } for r in ratings_rows])

        if ratings_df.empty:
            print("[pipeline] No ratings found – skipping ML pipeline.")
            return {}

        # Drop rows where street_name is null (shouldn't happen after seeding)
        ratings_df = ratings_df.dropna(subset=["street_name"])

        # ── Load streets from DB ──────────────────────────────────────────────
        streets_rows = Street.query.all()
        streets_df = pd.DataFrame([{
            "street_name":           s.name,
            "latitude":              s.latitude,
            "longitude":             s.longitude,
            "crime_rate_normalised": s.crime_rate_normalised,
            "lighting_score":        s.lighting_score,
            "avg_user_score":        s.avg_user_score,
            "composite_safety_score": s.composite_safety_score,
        } for s in streets_rows])

        print("\n=== SafeRoutes ML Pipeline ===")

        # ── Step 1: ALS collaborative filtering ──────────────────────────────
        print("\n[ALS] Running collaborative filtering...")
        als = ALSCollaborativeFilter(
            n_factors=config.ALS_N_FACTORS,
            n_iter=config.ALS_N_ITER,
            reg=config.ALS_REGULARISE,
        )
        als.fit(ratings_df)
        als_scores = als.get_street_scores()   # {street_name: score_0_1}

        # ── Step 2: Bayesian estimation ───────────────────────────────────────
        print("\n[Bayes] Computing Bayesian posterior scores...")
        bayes = BayesianSafetyEstimator(
            city_avg_safety=0.6,
            prior_strength=5.0,
        )
        bayes_scores = bayes.estimate_all_streets(ratings_df)

        # ── Step 3: Compute composite safety scores ───────────────────────────
        # composite = w1 * bayesian_score
        #           + w2 * (1 - crime_rate)
        #           + w3 * lighting_score
        w1, w2, w3 = config.SCORE_W1, config.SCORE_W2, config.SCORE_W3
        print("\n[composite] Updating composite scores in DB...")
        for street in streets_rows:
            b_score = bayes_scores.get(street.name, street.avg_user_score)
            composite = (
                w1 * b_score
                + w2 * (1.0 - street.crime_rate_normalised)
                + w3 * street.lighting_score
            )
            street.composite_safety_score = float(np.clip(composite, 0.0, 1.0))
            street.avg_user_score = float(np.clip(b_score, 0.0, 1.0))
        db_instance.session.commit()

        # ── Step 4: Reload streets_df with updated composite scores ──────────
        streets_df["composite_safety_score"] = [
            s.composite_safety_score for s in streets_rows
        ]

        # ── Step 5: Train sklearn Gradient Boosting model ─────────────────────
        print("\n[sklearn] Training Gradient Boosting safety predictor...")
        predictor = SafetyScorePredictor()
        metrics = predictor.train(ratings_df, streets_df)

        os.makedirs(config.DATA_DIR, exist_ok=True)
        predictor.save(config.SKLEARN_MODEL_PATH)

        print("\n=== Pipeline complete ===\n")
        return {
            "als":       als,
            "bayes":     bayes,
            "predictor": predictor,
            "metrics":   metrics,
        }
