"""
Configuration settings for SafeRoutes.
"""
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Flask secret key for session management
    SECRET_KEY = os.environ.get("SECRET_KEY", "saferoutes-dev-secret-2024")

    # SQLite database in the project root
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'saferoutes.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Paths for persisted model/graph artefacts
    DATA_DIR         = os.path.join(BASE_DIR, "data")
    GRAPH_PATH       = os.path.join(DATA_DIR, "edinburgh_graph.graphml")
    RIVER_MODEL_PATH = os.path.join(DATA_DIR, "river_model.pkl")
    SKLEARN_MODEL_PATH = os.path.join(DATA_DIR, "sklearn_model.pkl")

    # Edinburgh bounding box for OSMnx download
    # Covers central Edinburgh (Old Town → Stockbridge → Leith → Holyrood etc.)
    OSM_BBOX = {
        "north":  55.970,
        "south":  55.930,
        "east":  -3.140,
        "west":  -3.230,
    }

    # Bayesian composite score weights
    # w1 * bayesian_user_score + w2 * (1-crime_rate) + w3 * lighting_score
    SCORE_W1 = 0.5   # user perception weight
    SCORE_W2 = 0.3   # crime data weight
    SCORE_W3 = 0.2   # lighting quality weight

    # ALS hyper-parameters
    ALS_N_FACTORS   = 10   # number of latent factors
    ALS_N_ITER      = 20   # ALS iterations
    ALS_REGULARISE  = 0.01 # L2 regularisation lambda

    # Routing defaults
    DEFAULT_ALPHA = 0.7   # blend towards safe route by default
