# SafeRoutes — Full Stack & Machine Learning Explained
### A Beginner-Friendly Guide

---

## PART 1: WHAT IS SAFEROUTES?

SafeRoutes is a web application that helps university students in Edinburgh find safer
walking routes. Instead of just finding the *shortest* route (like Google Maps), it finds
the *safest* route by analysing:

- **Crowdsourced ratings** — real users rating how safe streets feel
- **Crime statistics** — crime rate data per neighbourhood
- **Lighting quality** — how well-lit streets are at night
- **Machine learning** — algorithms that predict safety scores and learn over time

---

## PART 2: THE FULL TECH STACK

Think of the tech stack as layers, like a cake. Each layer does a specific job.

```
┌─────────────────────────────────────────────────────┐
│  FRONTEND (what the user sees)                      │
│  HTML templates + Leaflet.js (interactive maps)     │
├─────────────────────────────────────────────────────┤
│  WEB FRAMEWORK (the traffic controller)             │
│  Flask — receives requests, calls the right code,   │
│  and sends back responses                           │
├─────────────────────────────────────────────────────┤
│  BUSINESS LOGIC (the brains)                        │
│  Python files in /models — safety scoring,          │
│  routing, machine learning                          │
├─────────────────────────────────────────────────────┤
│  DATABASE (memory)                                  │
│  SQLite + SQLAlchemy — stores users, ratings,       │
│  areas, and saved routes                            │
└─────────────────────────────────────────────────────┘
```

### Flask — The Web Framework

Flask is a Python library that lets you build websites. It listens for requests from
your browser and decides what code to run in response.

Example: when you go to `/api/route`, Flask runs the route-finding code and sends
back a list of coordinates.

```python
@app.route("/api/route", methods=["POST"])
def get_route():
    # This function runs when you visit /api/route
    data = request.get_json()      # Read the request data
    start_lat = data["start_lat"]  # Extract values
    ...
    return jsonify({"route": coordinates})  # Send back JSON
```

The `@app.route(...)` line is called a **decorator** — it tells Flask "when someone
visits this URL, run this function".

### SQLAlchemy — The Database Layer

SQLAlchemy lets Python talk to a database using Python objects instead of raw SQL.

A **model** is a Python class that maps to a database table:

```python
class SafetyRating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    safety_score = db.Column(db.Float, nullable=False)  # 1.0 to 5.0
    time_of_day = db.Column(db.String(20))              # "day", "evening", "night"
```

To save a rating:
```python
rating = SafetyRating(latitude=55.94, longitude=-3.19, safety_score=4.0)
db.session.add(rating)
db.session.commit()  # Actually writes to disk
```

### The Data Tables

The app stores four types of data:

| Table         | What it stores                                           |
|---------------|----------------------------------------------------------|
| User          | Usernames, emails, hashed passwords                      |
| SafetyRating  | Individual ratings submitted by users (1–5 scale)        |
| Area          | Edinburgh neighbourhoods with crime rates, lighting, etc.|
| Route         | Saved routes with coordinates and safety scores          |

---

## PART 3: THE MAP DATA — OSMnx & NetworkX

Before the app can find a route, it needs a map of Edinburgh's streets as a **graph**.

### What is a Graph?

A graph is a mathematical structure made of:
- **Nodes** — points (in this case, street intersections or endpoints)
- **Edges** — connections between nodes (the streets themselves)

```
[Node A] ——street——> [Node B] ——street——> [Node C]
```

### OSMnx

OSMnx downloads real street data from OpenStreetMap and converts it into a graph.

```python
import osmnx as ox

# Download Edinburgh's walking network
G = ox.graph_from_bbox(
    north=55.970, south=55.930,
    east=-3.140, west=-3.230,
    network_type="walk"   # Only walking streets, not motorways
)
```

This gives us a graph with thousands of nodes (street corners) and edges (streets).
Each edge stores its length in metres.

### NetworkX

NetworkX is the library that actually works with the graph — finding paths, reading
edge data, etc.

```python
import networkx as nx

# Find the nearest graph node to a GPS coordinate
origin_node = ox.nearest_nodes(G, X=start_lng, Y=start_lat)

# Find the safest path (more on this later)
path_nodes = nx.astar_path(G, origin_node, dest_node, weight="safety_weight")
```

---

## PART 4: MACHINE LEARNING — THE CORE IDEAS

Before diving into the specific algorithms, here are some concepts you'll see
throughout the code.

### What is Machine Learning?

Machine learning is teaching a computer to make predictions or decisions by showing
it examples, rather than writing explicit rules.

Example:
- **Rule-based:** "If crime rate > 0.7 AND it's night, safety = low"
- **ML-based:** "Here are 1,000 real ratings with their features — learn the pattern"

### Normalisation: Putting Numbers on the Same Scale

The app converts scores between two scales:

- **User-facing:** 1 to 5 (what you see on screen — like a star rating)
- **Model-facing:** 0 to 1 (what the maths uses internally)

```python
# Convert user rating (1–5) to model score (0–1)
norm_score = (score - 1) / 4.0
# Example: score=3 → (3-1)/4 = 0.5

# Convert back for display
display_score = norm_score * 4.0 + 1.0
```

This is important because ML algorithms work better when all inputs are on similar
scales. Mixing "latitude = 55.94" with "crime rate = 0.65" without scaling can cause
problems.

### Features

A **feature** is a piece of information the model uses to make a prediction.
SafeRoutes uses features like:

```
latitude, longitude, time_of_day, crime_rate, lighting_score,
number_of_ratings, average_historical_rating
```

These get assembled into a list (called a **feature vector**) and fed into the model.

---

## PART 5: MACHINE LEARNING ALGORITHM 1 — BAYESIAN SAFETY ESTIMATOR

**File:** `models/ml_pipeline.py`

### The Problem it Solves

Imagine an area has only 2 ratings: one 5-star, one 1-star. The average is 3. But
should we really trust that average from just 2 ratings? What if the next 10 ratings
are all 4-stars?

The Bayesian estimator handles this by saying: *"When we have little data, pull
the estimate towards the city average. As more data arrives, trust the data more."*

### The Beta Distribution (the Prior Belief)

In Bayesian statistics, we start with a **prior belief** before seeing any data.
SafeRoutes starts with:

```
City-wide average safety = 0.6   (our prior belief)
Prior strength = 5.0              (like having 5 fake votes to start)

alpha0 = 0.6 × 5.0 = 3.0  (pseudo "safe" votes)
beta0  = 0.4 × 5.0 = 2.0  (pseudo "unsafe" votes)
```

### Updating with Real Data

Each real rating is binarised:
- If normalised score ≥ 0.5 → it's a "safe" vote (success)
- If normalised score < 0.5  → it's an "unsafe" vote (failure)

```python
successes = sum(1 for s in scores if s >= 0.5)
failures  = len(scores) - successes

# Add real data to our prior beliefs
alpha_posterior = alpha0 + successes
beta_posterior  = beta0  + failures

# The estimated safety probability
safety_estimate = alpha_posterior / (alpha_posterior + beta_posterior)
```

### A Concrete Example

Area with 1 rating (score=4/5, normalised=0.75):

```
Before:  alpha=3.0, beta=2.0  →  estimate = 3/5 = 0.60
After:   alpha=4.0, beta=2.0  →  estimate = 4/6 = 0.67
```

Same area after 20 ratings (15 safe, 5 unsafe):

```
After:   alpha=18.0, beta=7.0  →  estimate = 18/25 = 0.72
```

With more data, the prior (city average) matters less and less. The estimate
converges towards the actual data.

### Why This is Useful

Without Bayesian estimation, a single 5-star rating in a new area would make it
look like the safest place in the city. The Bayesian approach prevents this by
being appropriately sceptical of small samples.

---

## PART 6: MACHINE LEARNING ALGORITHM 2 — COLLABORATIVE FILTERING (ALS)

**File:** `models/ml_pipeline.py`

### The Problem it Solves

Not all users rate areas the same way. Some users are harsh raters (they give low
scores everywhere), while others are generous. Collaborative filtering adjusts for
this by learning each user's rating *style*.

It answers: *"Given how User A rates areas, and how their tastes compare to User B,
what would User A rate an area they've never visited?"*

This is the same idea Netflix uses to recommend films.

### Matrix Factorisation

Imagine a grid where rows = users and columns = areas. Each cell holds a rating.
Most cells are empty (users haven't rated most areas):

```
           OldTown  NewTown  Leith  Cowgate
User Alice    4.0     3.5    ---     ---
User Bob      ---     4.0    2.5    3.0
User Carol    3.0     ---    ---     4.5
```

ALS (Alternating Least Squares) tries to *fill in the gaps* by finding hidden
patterns. It does this by creating two smaller matrices:

- **U** — what each user "cares about" (n_users × k factors)
- **V** — what each area "provides" (n_areas × k factors)

The idea: `Rating ≈ U[user] · V[area]` (dot product of their factor vectors)

```python
# Setup
als = ALSCollaborativeFilter(
    n_factors=10,    # k=10 hidden factors (like "nightlife", "quietness", etc.)
    n_iter=20,       # Repeat the training loop 20 times
    reg=0.01         # Regularisation (prevents overfitting — explained below)
)

als.fit(ratings_df)                # Train on historical ratings
als_scores = als.get_area_scores() # Get predicted safety per area
```

### The Alternating Least Squares Algorithm

The training alternates between two steps:

**Step 1:** Hold V fixed, update every row of U
**Step 2:** Hold U fixed, update every row of V
**Repeat**

For each update, it uses **ridge regression** — a mathematical formula that finds
the best fitting values while penalising overly large numbers (regularisation).

```python
# Update user factors (simplified):
A = V.T @ V + reg * identity_matrix   # V^T V + λI
b = V.T @ ratings_for_this_user
U[user] = solve(A, b)                 # Closed-form linear algebra solution
```

### Regularisation

The `reg=0.01` parameter is called **regularisation** (or the regularisation
coefficient, λ). It prevents the model from memorising the training data too
perfectly (overfitting).

Without regularisation, the model might learn "User Alice always rates Old Town 4.0"
exactly, but then fail to generalise to new users. With regularisation, it learns
smoother, more general patterns.

---

## PART 7: MACHINE LEARNING ALGORITHM 3 — GRADIENT BOOSTING (scikit-learn)

**File:** `models/ml_pipeline.py`

This is the most powerful standalone predictor in the app. It can predict a safety
score for *any* GPS coordinate, even ones no user has ever rated.

### What is Gradient Boosting?

Gradient Boosting builds many small decision trees, one after another. Each new tree
learns to fix the mistakes the previous trees made.

Think of it like hiring consultants one by one:
1. Consultant 1 makes rough predictions
2. Consultant 2 focuses on where Consultant 1 was wrong
3. Consultant 3 focuses on where 1 and 2 were wrong
4. ... and so on for 200 consultants

The final prediction is the sum of all their opinions.

### Decision Trees (the building blocks)

A decision tree splits data by asking yes/no questions:

```
Is crime rate > 0.6?
├── YES → Is it night?
│         ├── YES → Safety = 0.2 (very unsafe)
│         └── NO  → Safety = 0.45
└── NO  → Is lighting > 0.7?
          ├── YES → Safety = 0.8 (very safe)
          └── NO  → Safety = 0.6
```

Each tree in the ensemble is small (`max_depth=4` means at most 4 levels of splits).

### Using scikit-learn

scikit-learn follows a consistent pattern for all its models:

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. CREATE THE MODEL
model = GradientBoostingRegressor(
    n_estimators=200,   # 200 trees in the ensemble
    max_depth=4,        # Each tree can be at most 4 levels deep
    learning_rate=0.05, # How much each tree contributes (smaller = more careful)
    random_state=42     # Makes results reproducible (same result every run)
)

# 2. PREPARE FEATURES
# Time of day is text ("day", "evening", "night") — convert to numbers
encoder = LabelEncoder()
time_encoded = encoder.fit_transform(["day", "night", "evening"])
# Result: [0, 2, 1] — alphabetical order by default

# Feature matrix X: each row is one training example
X = np.array([
    [lat, lng, time_enc, crime_rate, lighting, num_ratings, avg_rating],
    # ... one row per data point
])
y = np.array([0.8, 0.3, 0.6, ...])  # Target: composite safety score

# 3. SPLIT DATA (keep some for testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # Use 20% for testing, 80% for training
    random_state=42
)

# 4. TRAIN
model.fit(X_train, y_train)

# 5. EVALUATE
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
# RMSE: average prediction error (lower is better)
# R²:   how much variance is explained (1.0 is perfect, 0 is useless)

# 6. PREDICT NEW POINTS
new_point = np.array([[55.95, -3.19, 2, 0.6, 0.5, 10, 0.6]])
predicted_score = model.predict(new_point)[0]
safe_score = np.clip(predicted_score, 0.0, 1.0)  # Keep between 0 and 1
```

### Saving and Loading Models

Training takes time. scikit-learn models can be saved to disk so you don't retrain
every time the server starts:

```python
import pickle

# Save
with open("data/sklearn_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load
with open("data/sklearn_model.pkl", "rb") as f:
    model = pickle.load(f)
```

---

## PART 8: MACHINE LEARNING ALGORITHM 4 — ONLINE LEARNING (River)

**File:** `models/online_learner.py`

### The Problem with Traditional ML

The scikit-learn Gradient Boosting model is trained once on all historical data.
To update it with new ratings, you'd have to retrain the whole model from scratch —
which takes time and computing power.

**Online learning** solves this: the model updates itself *instantly* each time a
new rating comes in, learning one example at a time.

### River

River is a Python library specifically designed for streaming/online machine learning.

```python
from river import ensemble, tree, preprocessing, metrics

# Build a pipeline: scale features, then predict with an ensemble
model = preprocessing.StandardScaler() | ensemble.BaggingRegressor(
    model=tree.HoeffdingAdaptiveTreeRegressor(
        grace_period=50,          # Wait for 50 samples before considering splits
        delta=1e-5,               # Confidence level for Hoeffding bound
        leaf_prediction="adaptive" # Automatically chooses best prediction type
    ),
    n_models=5,    # Ensemble of 5 trees (reduces variance)
    seed=42
)
```

### The Hoeffding Tree

A **Hoeffding Adaptive Tree** is a decision tree designed for streaming data.
The key idea is the **Hoeffding bound** — a mathematical theorem that tells you
how many samples you need to see before you can confidently say "Feature A is a
better split than Feature B".

This means the tree can make decisions about its structure without storing all
the data — it only needs running statistics.

### StandardScaler (Online Normalisation)

The pipeline starts with `StandardScaler`, which normalises features. Unlike
scikit-learn's version (which needs the full dataset first), River's version
updates the mean and variance *as new data arrives*:

```python
# Internally tracks running mean and variance
# After each observation, updates: mean = mean + (x - mean) / n
```

### The Online Learning API

River has a simple, consistent API — just two methods per model:

```python
# Define features as a Python dictionary
x = {
    "latitude":   55.947,
    "longitude":  -3.200,
    "time_day":   1.0,    # 1=yes, 0=no (one-hot encoded)
    "time_eve":   0.0,
    "time_ngt":   0.0,
    "crime_rate": 0.65,
    "lighting":   0.70,
    "hist_avg":   0.55    # Historical average rating for nearby area
}
y = 0.75  # True safety score (normalised 0–1)

# PREDICT before learning (so the metric is fair)
y_pred = model.predict_one(x)

# LEARN from the true value
model.learn_one(x, y)

# Track the running error
mae = metrics.MAE()  # Mean Absolute Error
mae.update(y, y_pred)
print(f"Current MAE: {mae.get():.3f}")
```

The key difference from scikit-learn: `predict_one` and `learn_one` process
**one sample at a time**, making it suitable for real-time updates.

### How it Fits in the App

When a user submits a rating:
1. The River model immediately makes a prediction (before learning)
2. The model learns from the true rating
3. The updated model is saved to disk
4. All future predictions benefit from this new data

```python
# In app.py, after saving a rating to the database:
y_pred = _online_learner.learn_one(
    lat=rating.latitude,
    lng=rating.longitude,
    time_of_day=rating.time_of_day,
    crime_rate=area.crime_rate_normalised,
    lighting=area.lighting_score,
    hist_avg=area.avg_user_score,
    true_score=norm_score   # The actual rating submitted
)
_online_learner.save(Config.RIVER_MODEL_PATH)
```

---

## PART 9: THE COMPOSITE SAFETY SCORE

All the ML algorithms feed into a single composite score used for routing.

### Formula

```python
composite = (0.5 * bayesian_score        # What users say (50% weight)
           + 0.3 * (1.0 - crime_rate)   # Crime data (30% weight, inverted)
           + 0.2 * lighting_score)       # Lighting (20% weight)

composite = np.clip(composite, 0.0, 1.0) # Ensure it stays between 0 and 1
```

The weights (0.5, 0.3, 0.2) are set in `config.py` and reflect how much each
factor should influence the final score.

Note that `crime_rate` is *inverted* (`1.0 - crime_rate`) because a higher crime
rate should produce a *lower* safety score.

---

## PART 10: THE ROUTING ALGORITHM

**File:** `models/route_engine.py`

### The Problem

Finding the safest route is not the same as finding the shortest route. A direct
but dangerous alley might be "shortest" by distance, but a longer route through
well-lit streets is safer.

SafeRoutes balances both with a weighted cost function.

### Edge Weights

Each street (edge in the graph) gets a **weight** — a number representing its
"cost". Lower cost = preferred by the algorithm.

```python
def compute_edge_weight(edge_data, alpha=0.7):
    dist   = edge_data["length"]         # Street length in metres
    safety = edge_data["safety_score"]   # ML-predicted score (0–1)

    # Cost if you only cared about safety (distance / safety = penalise unsafe)
    safety_cost   = dist / safety

    # Cost if you only cared about distance
    distance_cost = dist

    # Blend the two: alpha=0.7 means 70% safety, 30% distance
    return alpha * safety_cost + (1 - alpha) * distance_cost
```

If `safety = 0.1` (very unsafe), `safety_cost = dist / 0.1 = 10 × dist` — the
algorithm really doesn't want to use this street.

If `safety = 0.9` (very safe), `safety_cost = dist / 0.9 ≈ 1.1 × dist` — nearly
the same as the raw distance, so this street is preferred.

### A* Algorithm

SafeRoutes uses A* (pronounced "A-star") — a classic pathfinding algorithm.

A* is smarter than a brute-force search because it uses a **heuristic** — an
estimate of the remaining distance to the goal. It prioritises paths that seem
closer to the destination, dramatically speeding up the search.

The heuristic used here is the **haversine distance** — the straight-line distance
between two GPS coordinates on a sphere (Earth).

```python
import math

def _haversine_distance(node1, node2, G):
    lat1, lng1 = G.nodes[node1]["y"], G.nodes[node1]["x"]
    lat2, lng2 = G.nodes[node2]["y"], G.nodes[node2]["x"]

    R = 6_371_000  # Earth's radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)

    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

# Use A* with our safety-weighted edges and haversine heuristic
path_nodes = nx.astar_path(
    G,
    source=origin_node,
    target=dest_node,
    heuristic=_haversine_distance,
    weight=compute_edge_weight  # Uses the function defined above
)
```

Alternatively, users can choose **Dijkstra's algorithm**, which guarantees the
optimal path but explores more nodes (slower, but always correct):

```python
path_nodes = nx.dijkstra_path(G, origin_node, dest_node, weight=compute_edge_weight)
```

### Colour-Coding the Route

Once the path is found, each street segment is colour-coded for display:

```python
for node_a, node_b in zip(path_nodes, path_nodes[1:]):
    edge_data = G[node_a][node_b][0]
    safety = edge_data["safety_score"]

    if safety >= 0.7:
        colour = "#22c55e"   # Green — safe
    elif safety >= 0.4:
        colour = "#eab308"   # Yellow — moderate
    else:
        colour = "#ef4444"   # Red — avoid
```

---

## PART 11: HOW ALL THE ML PIECES FIT TOGETHER

Here is the complete lifecycle of a prediction:

### At Server Startup

```
1. Load historical ratings from the database
2. Train the Bayesian estimator → compute area scores
3. Train the ALS collaborative filter → refine area scores with user patterns
4. Update composite_safety_score for all areas in the database
5. Train the scikit-learn Gradient Boosting model on the combined data
6. Warm up the River online learner by replaying historical ratings
7. Save all models to disk (data/ folder)
```

### When a User Submits a Rating

```
New rating arrives (lat, lng, score, time_of_day)
         ↓
1. Save to database (SafetyRating table)
2. Find nearest Edinburgh neighbourhood
3. River model: learn_one() — instant update, no retraining
4. Recompute Bayesian posterior for that neighbourhood
5. Update composite_safety_score for that neighbourhood in DB
```

### When a User Requests a Route

```
Route request (start, end, alpha)
         ↓
1. Load Edinburgh street graph (from disk or download from OSM)
2. For every street in the graph:
   scikit-learn model: predict_score(midpoint lat/lng, time_of_day)
   → store as edge["safety_score"]
3. Compute edge weights: blend safety and distance using alpha
4. Run A* or Dijkstra to find the lowest-weight path
5. Colour-code each segment (green/yellow/red)
6. Return route coordinates + safety summary to the frontend
```

---

## PART 12: KEY LIBRARIES — QUICK REFERENCE

| Library       | What it does in this app                                          |
|---------------|-------------------------------------------------------------------|
| Flask         | Web server framework — handles HTTP requests and routing          |
| SQLAlchemy    | Database ORM — lets Python talk to SQLite without raw SQL         |
| scikit-learn  | Gradient Boosting model — predicts safety scores from features    |
| River         | Online learning — updates instantly from new user ratings         |
| NumPy         | Numerical computing — matrix maths, array operations              |
| Pandas        | Data manipulation — loads and processes tabular data (DataFrames) |
| OSMnx         | Downloads Edinburgh street map from OpenStreetMap                 |
| NetworkX      | Graph algorithms — A*, Dijkstra, edge/node manipulation           |
| Werkzeug      | Password hashing for secure login                                 |

---

## PART 13: COMMON PATTERNS IN THE CODE

### np.clip — Keeping Values in Range

```python
score = np.clip(score, 0.0, 1.0)
# If score = 1.3 → becomes 1.0
# If score = -0.2 → becomes 0.0
# If score = 0.7 → stays 0.7
```

### DataFrame Operations (Pandas)

```python
# Group by area and time, compute statistics
agg_data = ratings_df.groupby(["area_name", "time_of_day"]).agg(
    num_ratings=("safety_score", "count"),
    avg_rating=("safety_score", "mean")
)

# Merge two tables on a shared column
merged = agg_data.merge(areas_df, on="area_name")

# Filter rows
night_ratings = ratings_df[ratings_df["time_of_day"] == "night"]

# Access a column
scores = ratings_df["safety_score"].values  # Returns NumPy array
```

### Matrix Maths (NumPy)

```python
import numpy as np

# Dot product (used in ALS prediction)
score = U[user_idx] @ V[area_idx]  # @ is the matrix multiply operator

# Matrix multiply
R_predicted = U @ V.T  # (n_users, k) × (k, n_areas) = (n_users, n_areas)

# Mean across a dimension
area_scores = np.mean(R_predicted, axis=0)  # Average across users (rows)

# Solve linear system Ax = b (used in ALS solver)
x = np.linalg.solve(A, b)
```

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 80% training, 20% testing
    random_state=42   # Reproducible split
)
```

This is a fundamental practice: never evaluate a model on the same data it
trained on. The test set simulates "new, unseen" data.

---

## SUMMARY

SafeRoutes is a safety-first routing app that combines several layers of
machine learning working together:

1. **Bayesian Estimation** — smart averaging that handles sparse data gracefully
2. **Collaborative Filtering (ALS)** — learns user rating patterns like Netflix
3. **Gradient Boosting (scikit-learn)** — powerful offline predictor for any location
4. **Online Learning (River)** — real-time updates from new user submissions
5. **Graph Routing (NetworkX + OSMnx)** — finds safest path on real Edinburgh streets

The models are trained on seed data at startup, then continuously refined as
users submit ratings. Each new rating immediately improves the River model and
updates the Bayesian estimate, meaning the app gets smarter the more people use it.
