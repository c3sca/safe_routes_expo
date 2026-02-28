"""
ml_pipeline.py – Core Machine Learning pipeline for SafeRoutes.

Contains three main components:
  1. ALS collaborative filtering  – calibrate user rating biases
  2. Bayesian score estimation    – robust posterior safety scores
  3. Scikit-learn Random Forest   – predict safety for any map point

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
      We have a sparse rating matrix  R  (users × areas).
      R[u, a] is the safety score user u gave to area a (or missing).

      ALS factorises R ≈ U · Vᵀ, where:
        U  ∈ ℝ^{n_users  × k}   – user latent factors
        V  ∈ ℝ^{n_areas × k}   – area latent factors
        k  = n_factors

      We alternate between:
        • Fixing V, solving for each row u of U analytically (ridge regression):
              u_i = (Vᵀ Cᵢ V + λI)⁻¹ Vᵀ Cᵢ r_i
          where Cᵢ is a diagonal confidence matrix (1 where rated, 0 elsewhere)
          and r_i is user i's ratings vector.
        • Fixing U, solving for each row v of V analogously.

      λ (lambda) is L2 regularisation to prevent overfitting.

    After training, predicted rating for (user u, area a) = U[u] · V[a].
    """

    def __init__(self, n_factors: int = 10, n_iter: int = 20, reg: float = 0.01):
        self.n_factors = n_factors
        self.n_iter    = n_iter
        self.reg       = reg          # λ regularisation
        self.U         = None         # user factor matrix
        self.V         = None         # area factor matrix
        self.user_index  = {}         # user_id  → row index in R
        self.area_index  = {}         # area_name → col index in R
        self.area_names  = []

    def _build_matrix(self, ratings_df: pd.DataFrame) -> np.ndarray:
        """
        Convert a ratings DataFrame into a dense user×area matrix.

        ratings_df columns: user_id, area_name, safety_score (1–5)
        Missing entries are 0 (treated as unobserved, not as "zero safety").
        """
        # Build index maps
        users = ratings_df["user_id"].unique().tolist()
        areas = ratings_df["area_name"].unique().tolist()
        self.user_index = {uid: i for i, uid in enumerate(users)}
        self.area_index = {a: j   for j, a   in enumerate(areas)}
        self.area_names = areas

        n_users = len(users)
        n_areas = len(areas)
        R = np.zeros((n_users, n_areas))

        for _, row in ratings_df.iterrows():
            u = self.user_index[row["user_id"]]
            a = self.area_index[row["area_name"]]
            # Normalise score 1–5 → 0–1
            R[u, a] = (row["safety_score"] - 1) / 4.0

        return R

    def _solve_row(self, fixed: np.ndarray, R_slice: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Solve for a single row of the factor matrix analytically.

        For user i (or area j), with the other factor matrix Fixed (V or U):
          min_x  Σ_{observed j} (R[i,j] - x · Fixed[j])²  +  λ ||x||²

        Closed-form solution (ordinary ridge regression):
          x = (Fᵀ C F + λ I)⁻¹ Fᵀ C r

        where:
          F = Fixed[mask]        – rows of Fixed where ratings exist
          C = identity           – uniform confidence for all observed ratings
          r = R_slice[mask]      – observed ratings for this row/column

        Parameters
        ----------
        fixed   : (n_items, k) – the "other" factor matrix (V when updating U)
        R_slice : (n_items,)   – one row/column of the rating matrix
        mask    : (n_items,)   – boolean, True where rating is observed
        """
        F = fixed[mask]              # (n_obs, k) observed factor rows
        r = R_slice[mask]            # (n_obs,)   observed ratings

        if F.shape[0] == 0:
            # No observations – return zero vector
            return np.zeros(self.n_factors)

        k = self.n_factors
        # Fᵀ F  +  λ I   →  (k, k)
        A = F.T @ F + self.reg * np.eye(k)
        # Fᵀ r            →  (k,)
        b = F.T @ r
        # Solve A x = b
        return np.linalg.solve(A, b)

    def fit(self, ratings_df: pd.DataFrame) -> None:
        """Train ALS on a DataFrame of (user_id, area_name, safety_score)."""
        R = self._build_matrix(ratings_df)
        n_users, n_areas = R.shape
        k = self.n_factors

        # Initialise factor matrices with small random values
        rng = np.random.default_rng(42)
        self.U = rng.normal(0, 0.1, (n_users, k))
        self.V = rng.normal(0, 0.1, (n_areas, k))

        # Observation mask: True where R[u,a] > 0 (i.e., rating was given)
        obs = R > 0

        errors = []
        for iteration in range(self.n_iter):
            # ── Step A: fix V, update every row of U ──────────────────────
            for u in range(n_users):
                self.U[u] = self._solve_row(self.V, R[u, :], obs[u, :])

            # ── Step B: fix U, update every row of V ──────────────────────
            for a in range(n_areas):
                self.V[a] = self._solve_row(self.U, R[:, a], obs[:, a])

            # Track reconstruction error on observed entries
            R_pred = self.U @ self.V.T
            err = np.sqrt(np.mean((R[obs] - R_pred[obs]) ** 2))
            errors.append(err)

        print(f"[ALS] Final reconstruction RMSE (observed): {errors[-1]:.4f}")

    def predict(self, user_id: int, area_name: str) -> float:
        """
        Predict the normalised safety score (0–1) for a (user, area) pair.
        Falls back to the area's average latent score if user is unknown.
        """
        a_idx = self.area_index.get(area_name)
        if a_idx is None:
            return 0.5   # completely unknown area

        if user_id in self.user_index:
            u_idx = self.user_index[user_id]
            score = float(self.U[u_idx] @ self.V[a_idx])
        else:
            # Unknown user – use mean of all user factors as proxy
            score = float(np.mean(self.U, axis=0) @ self.V[a_idx])

        # Clamp to [0, 1]
        return float(np.clip(score, 0.0, 1.0))

    def get_area_scores(self) -> dict:
        """
        Return ALS-derived latent safety score for every area.
        Score = mean predicted rating across all users.
        Returns dict: area_name → score (0–1).
        """
        # Predict ratings for every user–area pair, then average across users
        R_pred = self.U @ self.V.T          # (n_users, n_areas)
        area_scores = np.clip(np.mean(R_pred, axis=0), 0, 1)
        return {name: float(area_scores[j])
                for name, j in self.area_index.items()}


# ──────────────────────────────────────────────────────────────
# 2.  BAYESIAN SAFETY SCORE ESTIMATOR
# ──────────────────────────────────────────────────────────────

class BayesianSafetyEstimator:
    """
    Bayesian estimation of area safety using a Beta–Bernoulli conjugate model.

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
        """
        Parameters
        ----------
        city_avg_safety : float
            Prior belief about how safe the city is on average (0–1).
        prior_strength : float
            How many pseudo-observations the prior is worth.
            Higher = prior has more influence; lower = data dominates quickly.
        """
        self.city_avg  = city_avg_safety
        self.strength  = prior_strength
        # Prior hyper-parameters
        self.alpha0 = city_avg_safety * prior_strength
        self.beta0  = (1.0 - city_avg_safety) * prior_strength

    def estimate(self, normalised_scores: list) -> float:
        """
        Compute the posterior expected safety probability for one area.

        Parameters
        ----------
        normalised_scores : list of float
            User ratings for this area, already normalised to [0, 1].

        Returns
        -------
        float : posterior mean safety score in [0, 1]
        """
        # Count "successes" (score ≥ 0.5 → rated as safe)
        successes = sum(1 for s in normalised_scores if s >= 0.5)
        failures  = len(normalised_scores) - successes

        # Posterior hyper-parameters (conjugate Beta update)
        alpha_post = self.alpha0 + successes
        beta_post  = self.beta0  + failures

        # Posterior mean of Beta distribution: α / (α + β)
        return alpha_post / (alpha_post + beta_post)

    def estimate_all_areas(self, ratings_df: pd.DataFrame) -> dict:
        """
        Compute Bayesian posterior safety scores for all areas.

        Parameters
        ----------
        ratings_df : DataFrame with columns [area_name, safety_score]
            safety_score is on the 1–5 scale.

        Returns
        -------
        dict : area_name → bayesian_score (0–1)
        """
        results = {}
        for area_name, group in ratings_df.groupby("area_name"):
            # Normalise scores 1–5 → 0–1
            normalised = [(s - 1) / 4.0 for s in group["safety_score"]]
            results[area_name] = self.estimate(normalised)
        return results


# ──────────────────────────────────────────────────────────────
# 3.  SCIKIT-LEARN GRADIENT BOOSTING SAFETY PREDICTOR
# ──────────────────────────────────────────────────────────────

class SafetyScorePredictor:
    """
    Gradient Boosting Regressor trained on area seed data.

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
        self.le_time = LabelEncoder()   # encodes 'day'/'evening'/'night' → int
        self.trained = False

    def _build_features(self, ratings_df: pd.DataFrame,
                        areas_df: pd.DataFrame) -> tuple:
        """
        Join ratings to area data and build the feature matrix X and target y.

        Returns (X: np.ndarray, y: np.ndarray)
        """
        # Aggregate ratings per (area_name, time_of_day)
        agg = (
            ratings_df
            .groupby(["area_name", "time_of_day"])
            .agg(num_ratings=("safety_score", "count"),
                 avg_rating=("safety_score", "mean"))
            .reset_index()
        )
        # Normalise avg_rating 1–5 → 0–1
        agg["avg_rating_norm"] = (agg["avg_rating"] - 1) / 4.0

        # Merge with area geographic / crime / lighting data
        merged = agg.merge(areas_df, on="area_name", how="left")

        # Encode time_of_day as integer (fit on full data to capture all classes)
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

    def train(self, ratings_df: pd.DataFrame, areas_df: pd.DataFrame) -> dict:
        """
        Train the Gradient Boosting model on seed data.

        Parameters
        ----------
        ratings_df : DataFrame  [user_id, area_name, safety_score, time_of_day]
        areas_df   : DataFrame  [area_name, latitude, longitude,
                                 crime_rate_normalised, lighting_score,
                                 composite_safety_score]

        Returns
        -------
        dict with 'rmse' and 'r2' on the held-out test set.
        """
        X, y = self._build_features(ratings_df, areas_df)

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
            # Fallback heuristic: weight crime and lighting equally
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
    from models.database import SafetyRating, Area

    with app.app_context():
        # ── Load ratings from DB ──────────────────────────────────────────
        ratings_rows = SafetyRating.query.all()
        ratings_df = pd.DataFrame([{
            "user_id":      r.user_id,
            "area_name":    r.area_name,
            "safety_score": r.safety_score,
            "time_of_day":  r.time_of_day,
            "latitude":     r.latitude,
            "longitude":    r.longitude,
        } for r in ratings_rows])

        if ratings_df.empty:
            print("[pipeline] No ratings found – skipping ML pipeline.")
            return {}

        # ── Load areas from DB ────────────────────────────────────────────
        areas_rows = Area.query.all()
        areas_df = pd.DataFrame([{
            "area_name":             a.name,
            "latitude":              a.latitude,
            "longitude":             a.longitude,
            "crime_rate_normalised": a.crime_rate_normalised,
            "lighting_score":        a.lighting_score,
            "avg_user_score":        a.avg_user_score,
            "composite_safety_score": a.composite_safety_score,
        } for a in areas_rows])

        print("\n=== SafeRoutes ML Pipeline ===")

        # ── Step 1: ALS collaborative filtering ──────────────────────────
        print("\n[ALS] Running collaborative filtering...")
        als = ALSCollaborativeFilter(
            n_factors=config.ALS_N_FACTORS,
            n_iter=config.ALS_N_ITER,
            reg=config.ALS_REGULARISE,
        )
        als.fit(ratings_df)
        als_scores = als.get_area_scores()   # {area_name: score_0_1}

        # ── Step 2: Bayesian estimation ───────────────────────────────────
        print("\n[Bayes] Computing Bayesian posterior scores...")
        bayes = BayesianSafetyEstimator(
            city_avg_safety=0.6,
            prior_strength=5.0,
        )
        bayes_scores = bayes.estimate_all_areas(ratings_df)

        # ── Step 3: Compute composite safety scores ───────────────────────
        # composite = w1 * bayesian_score
        #           + w2 * (1 - crime_rate)
        #           + w3 * lighting_score
        w1, w2, w3 = config.SCORE_W1, config.SCORE_W2, config.SCORE_W3
        print("\n[composite] Updating composite scores in DB...")
        for area in areas_rows:
            b_score = bayes_scores.get(area.name, area.avg_user_score)
            composite = (
                w1 * b_score
                + w2 * (1.0 - area.crime_rate_normalised)
                + w3 * area.lighting_score
            )
            area.composite_safety_score = float(np.clip(composite, 0.0, 1.0))
            # Also store the Bayesian score as avg_user_score
            area.avg_user_score = float(np.clip(b_score, 0.0, 1.0))
        db_instance.session.commit()

        # ── Step 4: Reload areas_df with updated composite scores ─────────
        areas_df["composite_safety_score"] = [
            a.composite_safety_score for a in areas_rows
        ]

        # ── Step 5: Train sklearn Gradient Boosting model ─────────────────
        print("\n[sklearn] Training Gradient Boosting safety predictor...")
        predictor = SafetyScorePredictor()
        metrics = predictor.train(ratings_df, areas_df)

        # Save model to disk
        os.makedirs(config.DATA_DIR, exist_ok=True)
        predictor.save(config.SKLEARN_MODEL_PATH)

        print("\n=== Pipeline complete ===\n")
        return {
            "als":       als,
            "bayes":     bayes,
            "predictor": predictor,
            "metrics":   metrics,
        }
