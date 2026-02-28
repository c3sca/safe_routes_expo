"""
route_engine.py – Edinburgh street graph loading and safety-optimised routing.

Two routing algorithms are provided:
  • A* (A-star)    – uses a heuristic (geographic distance to goal) to find
                     the optimal path faster than Dijkstra on large graphs.
  • Dijkstra       – exhaustive shortest-path; guaranteed optimal but slower.

Both algorithms operate on a safety-weighted graph where each edge's cost is:

  cost = alpha * safety_cost + (1 - alpha) * distance_cost

where:
  safety_cost   = distance_m / safety_score   (safer edges are cheaper)
  distance_cost = distance_m                  (raw physical distance)
  alpha ∈ [0, 1]: 0 = pure shortest path, 1 = pure safest path

The street graph is downloaded from OpenStreetMap via OSMnx and cached locally
as a .graphml file so subsequent starts don't need a network call.
"""
import os
import json
import math
import pickle
from typing import Optional, Tuple, List

import networkx as nx
import numpy as np

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("[OSMnx] osmnx not installed – routing will use fallback mode.")


# ──────────────────────────────────────────────────────────────
# GRAPH LOADING
# ──────────────────────────────────────────────────────────────

def load_edinburgh_graph(graph_path: str, bbox: dict) -> Optional[object]:
    """
    Load the Edinburgh walkable street graph.

    First checks for a locally cached .graphml file. If not found, downloads
    from OpenStreetMap via OSMnx and saves the cache.

    Parameters
    ----------
    graph_path : str
        Path to the local .graphml cache file.
    bbox : dict
        Bounding box with keys: north, south, east, west.

    Returns
    -------
    networkx.MultiDiGraph or None
    """
    if not OSMNX_AVAILABLE:
        print("[OSMnx] Cannot load graph: osmnx not available.")
        return None

    # ── Use cached graph if it exists ───────────────────────────────────
    if os.path.exists(graph_path):
        print(f"[OSMnx] Loading cached graph from {graph_path}...")
        try:
            G = ox.load_graphml(graph_path)
            print(f"[OSMnx] Graph loaded: {G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges.")
            return G
        except Exception as e:
            print(f"[OSMnx] Cache load failed ({e}), re-downloading...")

    # ── Download from OSM ────────────────────────────────────────────────
    print("[OSMnx] Downloading Edinburgh walkable street graph...")
    print("        (This may take 30–60 seconds on first run)")
    try:
        G = ox.graph_from_bbox(
            north=bbox["north"],
            south=bbox["south"],
            east=bbox["east"],
            west=bbox["west"],
            network_type="walk",
            simplify=True,
        )
        # Add edge bearings and lengths (useful for display)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)

        # Save to cache
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        ox.save_graphml(G, graph_path)
        print(f"[OSMnx] Graph saved to {graph_path}")
        print(f"[OSMnx] Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        return G

    except Exception as e:
        print(f"[OSMnx] Download failed: {e}")
        print("[OSMnx] Routing will be disabled until a graph is available.")
        return None


# ──────────────────────────────────────────────────────────────
# SAFETY WEIGHTS
# ──────────────────────────────────────────────────────────────

def annotate_graph_safety(G, predictor, time_of_day: str = "night") -> None:
    """
    Add a 'safety_score' attribute to every edge in the graph.

    For each edge we:
      1. Compute the midpoint coordinates (lat, lng).
      2. Call the sklearn predictor to get a safety score (0–1).
      3. Store that score as the edge attribute 'safety_score'.

    This is done once at startup so routing calls are fast.

    Parameters
    ----------
    G          : nx.MultiDiGraph from OSMnx
    predictor  : SafetyScorePredictor (from ml_pipeline.py)
    time_of_day: default time of day for scoring ('night' = conservative)
    """
    if G is None or predictor is None:
        return

    print(f"[routing] Annotating graph edges with safety scores ({time_of_day})...")
    node_coords = {}
    for node, data in G.nodes(data=True):
        node_coords[node] = (data.get("y", 55.95), data.get("x", -3.19))

    for u, v, key, data in G.edges(keys=True, data=True):
        # Midpoint of the edge
        lat_u, lng_u = node_coords.get(u, (55.95, -3.19))
        lat_v, lng_v = node_coords.get(v, (55.95, -3.19))
        mid_lat = (lat_u + lat_v) / 2.0
        mid_lng = (lng_u + lng_v) / 2.0

        # Predict safety at this midpoint
        score = predictor.predict_score(
            lat=mid_lat,
            lng=mid_lng,
            time_of_day=time_of_day,
        )
        # Ensure score is never zero to avoid division by zero
        data["safety_score"] = max(score, 0.01)

    print("[routing] Safety annotation complete.")


def compute_edge_weight(data: dict, distance_key: str,
                        alpha: float = 0.7) -> float:
    """
    Compute the blended routing cost for a single edge.

    cost = alpha * (distance / safety_score) + (1 - alpha) * distance

    Derivation:
      • If alpha=1: we only minimise distance/safety, strongly preferring safe paths.
      • If alpha=0: we only minimise distance, ignoring safety (shortest path).
      • The distance/safety term makes long, dangerous edges very expensive and
        short, safe edges cheap.

    Parameters
    ----------
    data         : edge attribute dict
    distance_key : name of the distance attribute in the graph (usually 'length')
    alpha        : safety preference [0, 1]

    Returns
    -------
    float : blended edge cost (in metres, conceptually)
    """
    dist  = float(data.get(distance_key, 100.0))          # metres
    safety = float(data.get("safety_score", 0.5))         # 0–1

    safety_cost   = dist / safety     # lower safety → higher cost
    distance_cost = dist

    return alpha * safety_cost + (1.0 - alpha) * distance_cost


# ──────────────────────────────────────────────────────────────
# PATHFINDING
# ──────────────────────────────────────────────────────────────

def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Great-circle distance in metres between two (lat, lng) points.
    Used as the A* heuristic – it is admissible (never overestimates)
    because we divide by a safety score ≤ 1 in the actual cost.
    """
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    return 2 * R * math.asin(math.sqrt(a))


def find_route(G, origin_lat: float, origin_lng: float,
               dest_lat: float, dest_lng: float,
               alpha: float = 0.7) -> Optional[dict]:
    """
    Find a safety-optimised walking route between two coordinates.

    Parameters
    ----------
    G           : OSMnx MultiDiGraph with 'safety_score' on edges
    origin_lat / origin_lng : start coordinates
    dest_lat   / dest_lng   : destination coordinates
    alpha       : 0 = shortest, 1 = safest (blended weight)

    Returns
    -------
    dict with keys:
      'coordinates' : list of [lat, lng] pairs for the route polyline
      'distance_km' : total route length in km
      'safety_score': average safety score along the route (0–1)
      'nodes'       : list of OSM node IDs
    or None if no route found.
    """
    if G is None:
        return None

    try:
        # Snap the start/end coordinates to the nearest graph nodes
        origin_node = ox.nearest_nodes(G, X=origin_lng, Y=origin_lat)
        dest_node   = ox.nearest_nodes(G, X=dest_lng,   Y=dest_lat)

        if origin_node == dest_node:
            return None

        # ── Define weight function ────────────────────────────────────────
        def weight_fn(u, v, data):
            # data is a dict of edge attributes (or a dict of dicts for multi-edges)
            # NetworkX passes a dict with all parallel-edge data for MultiDiGraph
            # We take the cheapest parallel edge.
            if isinstance(data, dict):
                # Single edge dict
                return compute_edge_weight(data, "length", alpha)
            # Multi-edge: find minimum cost parallel edge
            return min(compute_edge_weight(d, "length", alpha)
                       for d in data.values())

        # Dijkstra – guaranteed optimal, exhaustive search
        path_nodes = nx.dijkstra_path(
                G, origin_node, dest_node,
                weight=weight_fn,
            )

        # ── Extract coordinates and stats ─────────────────────────────────
        coordinates = []
        for node in path_nodes:
            node_data_item = G.nodes[node]
            coordinates.append([node_data_item["y"], node_data_item["x"]])

        # Compute total distance and average safety along the route
        total_dist_m = 0.0
        safety_scores = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            # Get edge data (first parallel edge)
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                # MultiDiGraph: edge_data is {0: {...}, 1: {...}, ...}
                best_edge = min(edge_data.values(), key=lambda d: d.get("length", 999))
                total_dist_m += best_edge.get("length", 0.0)
                safety_scores.append(best_edge.get("safety_score", 0.5))

        avg_safety = float(np.mean(safety_scores)) if safety_scores else 0.5

        return {
            "coordinates":  coordinates,
            "distance_km":  round(total_dist_m / 1000.0, 3),
            "safety_score": round(avg_safety, 3),
            "nodes":        path_nodes,
            "algorithm":    "dijkstra",
        }

    except nx.NetworkXNoPath:
        print(f"[routing] No path found between ({origin_lat},{origin_lng}) "
              f"and ({dest_lat},{dest_lng})")
        return None
    except Exception as e:
        print(f"[routing] Error during pathfinding: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# SAFETY COLOUR CODING FOR ROUTE SEGMENTS
# ──────────────────────────────────────────────────────────────

def colour_route_segments(G, path_nodes: list) -> list:
    """
    Return a list of colour-coded route segments for Leaflet.js display.

    Each segment is:
      { "coords": [[lat1,lng1],[lat2,lng2]], "color": "#hexcolor" }

    Colour scale:
      Green (#22c55e)  → safety ≥ 0.7
      Yellow (#eab308) → 0.4 ≤ safety < 0.7
      Red (#ef4444)    → safety < 0.4
    """
    segments = []
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        edge_data = G.get_edge_data(u, v)
        safety = 0.5
        if edge_data:
            best = min(edge_data.values(), key=lambda d: d.get("length", 999))
            safety = best.get("safety_score", 0.5)

        lat_u = G.nodes[u]["y"]
        lng_u = G.nodes[u]["x"]
        lat_v = G.nodes[v]["y"]
        lng_v = G.nodes[v]["x"]

        if safety >= 0.7:
            color = "#22c55e"   # green
        elif safety >= 0.4:
            color = "#eab308"   # yellow
        else:
            color = "#ef4444"   # red

        segments.append({
            "coords": [[lat_u, lng_u], [lat_v, lng_v]],
            "color": color,
            "safety": round(safety, 2),
        })

    return segments
