#!/usr/bin/env python3
"""
extract_streets.py
==================
One-time script to extract Edinburgh street/path data for the SafeRoutes
Street table.  Run this once to generate seed data before starting the app.

Outputs
-------
  data/edinburgh_streets.csv   – one row per named street, all attributes
  data/streets_list.py         – Python STREETS list to paste into seed_data.py

Data sources used (in priority order for crime data)
-----------------------------------------------------
  1. OSMnx              – street names, midpoints, geometry, lit tag, highway type
  2. Police Scotland API – real recorded crime counts per street name (free, no key)
  3. SIMD 2020          – official crime domain by datazone (optional, needs files)
  4. Neighbourhood heuristics – fallback using existing area-level data

Usage
-----
  Minimum (OSMnx + Police API + neighbourhood fallback):
    pip install osmnx requests numpy pandas
    python extract_streets.py

  Full (adds SIMD spatial join for better crime coverage):
    pip install osmnx geopandas requests numpy pandas shapely
    # Download SIMD files – see SIMD_DOWNLOAD_INSTRUCTIONS below
    python extract_streets.py

SIMD_DOWNLOAD_INSTRUCTIONS
--------------------------
  1. Go to: https://www.gov.scot/publications/scottish-index-of-multiple-deprivation-2020/
     Download "SIMD 2020v2 – indicator data" (CSV/Excel)
     Save as: data/simd_indicators.csv

  2. Go to: https://spatialdata.gov.scot/geonetwork/srv/api/records/1fc2c938-0446-4e88-8e45-d19bc2c13f9d
     Download the GeoJSON datazone boundaries
     Save as: data/simd_datazones.geojson

  The script detects these files automatically if present.
"""

import os
import sys
import time
import math
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import osmnx as ox
except ImportError:
    print("[!] osmnx is required:  pip install osmnx")
    sys.exit(1)

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_OK = True
    print("[✓] geopandas found – SIMD spatial join enabled if files are present")
except ImportError:
    GEOPANDAS_OK = False
    print("[i] geopandas not found – SIMD join skipped  (pip install geopandas to enable)")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
GRAPH_PATH  = DATA_DIR / "edinburgh_graph.graphml"
OUTPUT_CSV  = DATA_DIR / "edinburgh_streets.csv"
OUTPUT_PY   = DATA_DIR / "streets_list.py"
SIMD_CSV    = DATA_DIR / "simd_indicators.csv"       # optional
SIMD_ZONES  = DATA_DIR / "simd_datazones.geojson"    # optional

# ── Edinburgh bounding box (matches config.py) ────────────────────────────────
OSM_BBOX = dict(north=55.970, south=55.930, east=-3.140, west=-3.230)

# ── Neighbourhood fallback: name → (lat, lng, crime_rate, lighting_score) ─────
# These match your existing seed_data.py AREAS list
NEIGHBOURHOOD_FALLBACK = {
    "Old Town":       (55.9486, -3.1999, 0.65, 0.70),
    "New Town":       (55.9558, -3.1962, 0.30, 0.85),
    "Leith":          (55.9756, -3.1724, 0.60, 0.60),
    "The Meadows":    (55.9400, -3.1890, 0.25, 0.55),
    "Marchmont":      (55.9360, -3.1870, 0.15, 0.65),
    "Bruntsfield":    (55.9360, -3.2020, 0.20, 0.70),
    "Tollcross":      (55.9432, -3.2025, 0.50, 0.75),
    "Cowgate":        (55.9477, -3.1890, 0.75, 0.50),
    "Grassmarket":    (55.9471, -3.1955, 0.70, 0.60),
    "Lothian Road":   (55.9460, -3.2055, 0.68, 0.80),
    "Calton Hill":    (55.9553, -3.1790, 0.45, 0.40),
    "Haymarket":      (55.9460, -3.2190, 0.35, 0.80),
    "Morningside":    (55.9270, -3.2050, 0.10, 0.75),
    "Newington":      (55.9360, -3.1840, 0.20, 0.70),
    "Polwarth":       (55.9340, -3.2180, 0.15, 0.65),
    "Gorgie":         (55.9350, -3.2400, 0.45, 0.60),
    "Stockbridge":    (55.9590, -3.2130, 0.12, 0.80),
    "Dean Village":   (55.9540, -3.2210, 0.10, 0.50),
    "Fountainbridge": (55.9427, -3.2110, 0.40, 0.72),
    "Lauriston":      (55.9435, -3.1940, 0.35, 0.68),
}

# ── Highway type → baseline lighting heuristic ───────────────────────────────
_HIGHWAY_LIGHT = {
    "primary":       0.90,
    "secondary":     0.85,
    "tertiary":      0.80,
    "residential":   0.70,
    "living_street": 0.65,
    "unclassified":  0.60,
    "service":       0.55,
    "pedestrian":    0.75,
    "footway":       0.50,
    "path":          0.35,
    "track":         0.20,
    "steps":         0.45,
    "cycleway":      0.50,
}

# ── Surface → lighting adjustment ─────────────────────────────────────────────
_SURFACE_ADJ = {
    "cobblestone": -0.05,  # Old Town cobbles are darker
    "sett":        -0.05,
    "gravel":      -0.10,
    "dirt":        -0.15,
    "grass":       -0.20,
    "ground":      -0.15,
    "asphalt":      0.00,
    "concrete":     0.02,
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Extract street segments from OSMnx graph
# ══════════════════════════════════════════════════════════════════════════════

def _coerce_name(val) -> str | None:
    """Return a clean string street name, or None if unnamed."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, list):
        val = val[0] if val else None
        if val is None:
            return None
    s = str(val).strip()
    return s if s else None


def extract_segments(graph_path: Path, bbox: dict) -> pd.DataFrame:
    """
    Load (or download) the Edinburgh walkable graph and return a DataFrame
    with one row per named edge:
      name, mid_lat, mid_lng, length_m, highway, lit, surface
    """
    print("\n[1/5] Loading Edinburgh street graph...")

    if graph_path.exists():
        print(f"      Using cached graph: {graph_path}")
        G = ox.load_graphml(str(graph_path))
    else:
        print("      Downloading from OpenStreetMap (~30–90 s on first run)...")
        ox.settings.max_query_area_size = 1_000_000_000  # 1 000 km² – no subdivision for Edinburgh
        G = ox.graph_from_bbox(
            (bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
            network_type="walk",
            simplify=True,
        )
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(G, str(graph_path))
        print(f"      Saved graph cache → {graph_path}")

    print(f"      Graph: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")

    # Convert edges to GeoDataFrame
    _, edges_gdf = ox.graph_to_gdfs(G)

    records = []
    for _, row in edges_gdf.iterrows():
        name = _coerce_name(row.get("name"))
        if not name:
            continue  # skip unnamed alleys/connectors

        geom = row.geometry
        mp   = geom.interpolate(0.5, normalized=True)   # midpoint

        records.append({
            "name":     name,
            "mid_lat":  round(mp.y, 6),
            "mid_lng":  round(mp.x, 6),
            "length_m": float(row.get("length", 0.0)),
            "highway":  str(row.get("highway", "")),
            "lit":      str(row.get("lit", "")),
            "surface":  str(row.get("surface", "")),
        })

    df = pd.DataFrame(records)
    print(f"      {len(df):,} named segments → {df['name'].nunique():,} unique street names")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – Lighting score from OSM tags
# ══════════════════════════════════════════════════════════════════════════════

def compute_lighting(row: pd.Series) -> float:
    """
    Estimate lighting quality (0–1) for one segment.

    Priority:
      1. OSM `lit` tag       – explicit data (most accurate)
      2. Highway type        – heuristic based on road class
      3. Default 0.5         – when no information is available

    Surface type adjusts the result by a small delta.
    """
    lit = str(row.get("lit", "")).lower().strip()

    if lit in ("yes", "true", "1"):
        base = 0.90
    elif lit == "24/7":
        base = 0.95
    elif lit in ("sunset-sunrise", "automatic"):
        base = 0.85
    elif lit in ("no", "false", "0"):
        base = 0.15
    else:
        # Fall back to highway type heuristic
        hw = str(row.get("highway", "")).lower()
        # Edge case: OSMnx can store highway as "['residential', 'footway']"
        hw = hw.strip("[]' \"").split(",")[0].strip().strip("'\" ")
        base = _HIGHWAY_LIGHT.get(hw, 0.50)

    surf = str(row.get("surface", "")).lower()
    adj  = _SURFACE_ADJ.get(surf, 0.0)

    return float(np.clip(base + adj, 0.05, 0.99))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Police Scotland API crime data
# ══════════════════════════════════════════════════════════════════════════════

_POLICE_API = "https://data.police.uk/api/crimes-street/all-crime"


def _grid_points(bbox: dict, n: int = 3) -> list[tuple[float, float]]:
    """Return an n×n grid of (lat, lng) covering the bounding box."""
    lats = np.linspace(bbox["south"], bbox["north"], n)
    lngs = np.linspace(bbox["west"],  bbox["east"],  n)
    return [(float(lat), float(lng)) for lat in lats for lng in lngs]


def _recent_months(n: int = 6) -> list[str]:
    """
    Return the last n months as 'YYYY-MM' strings.
    Police API data lags ~3 months, so we start from 3 months ago.
    """
    months = []
    dt = datetime.utcnow() - timedelta(days=90)
    for _ in range(n):
        months.append(dt.strftime("%Y-%m"))
        dt -= timedelta(days=31)
    return months


def fetch_police_crimes(bbox: dict, n_months: int = 6) -> dict[str, int]:
    """
    Query the data.police.uk API for Edinburgh and return raw crime counts
    per street name.

    Strategy:
      • Use a 3×3 grid of lat/lng points across the bounding box
      • Query each month × grid point combination
      • De-duplicate by crime ID to avoid double-counting from overlapping radii
      • Strip "On or near " prefix from API street names
      • Return dict: street_name → crime_count
    """
    print("\n[2/5] Fetching Police Scotland crime data (data.police.uk)...")
    points = _grid_points(bbox, n=3)
    months = _recent_months(n_months)
    total_calls = len(points) * len(months)
    print(f"      {len(points)} grid points × {len(months)} months = {total_calls} API calls")
    print("      (each call has a 0.3 s delay to be polite to the API)")

    seen_ids: set = set()
    street_counts: dict[str, int] = defaultdict(int)
    failed = 0
    call_n = 0

    for month in months:
        for lat, lng in points:
            call_n += 1
            if call_n % 10 == 0:
                print(f"      ... {call_n}/{total_calls} calls done")
            try:
                resp = requests.get(
                    _POLICE_API,
                    params={"lat": lat, "lng": lng, "date": month},
                    timeout=20,
                )
                # 404 = no data for that month/location (normal for some months)
                if resp.status_code == 404:
                    time.sleep(0.3)
                    continue
                resp.raise_for_status()
                crimes = resp.json()
            except Exception as e:
                failed += 1
                time.sleep(0.3)
                continue

            for crime in crimes:
                # De-duplicate: use persistent_id if available, else id
                cid = crime.get("persistent_id") or crime.get("id") or ""
                if cid and cid in seen_ids:
                    continue
                if cid:
                    seen_ids.add(cid)

                raw_name = (crime.get("location", {})
                                 .get("street", {})
                                 .get("name", ""))
                if not raw_name:
                    continue

                # Clean: "On or near Princes Street" → "Princes Street"
                clean = raw_name.lower()
                for prefix in ("on or near ", "near ", "at ", "opposite "):
                    if clean.startswith(prefix):
                        clean = clean[len(prefix):]
                        break
                clean = clean.title().strip()
                if clean:
                    street_counts[clean] += 1

            time.sleep(0.3)

    total_crimes = sum(street_counts.values())
    print(f"      {len(seen_ids):,} unique crimes | "
          f"{len(street_counts):,} named streets | "
          f"{failed} failed requests")

    if not street_counts:
        print("      [!] No data returned – will fall back to neighbourhood heuristic")

    return dict(street_counts)


def normalise_crime_counts(counts: dict) -> dict[str, float]:
    """
    Scale raw crime counts to 0–1.
    Clips at the 95th percentile so a handful of very high-crime streets
    (e.g. Cowgate, Princes Street) don't compress everything else to zero.
    """
    if not counts:
        return {}
    vals = np.array(list(counts.values()), dtype=float)
    p95  = np.percentile(vals, 95) or 1.0
    return {name: float(np.clip(v / p95, 0.0, 1.0)) for name, v in counts.items()}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – SIMD 2020 spatial join (optional)
# ══════════════════════════════════════════════════════════════════════════════

def load_simd(simd_csv: Path, simd_zones: Path):
    """
    Load SIMD 2020 crime domain scores and return a GeoDataFrame of
    datazones with a 'crime_norm' column (0–1, higher = more crime).

    Returns None if files are missing or geopandas is unavailable.

    Expected SIMD indicator CSV columns (any SIMD 2020 release):
      Data_Zone  (e.g. "S01009247")
      One of: crime_rate, SIMD2020v2_Crime_crime_count,
              SIMD2020_Crime_crime_rate, etc.
    """
    if not GEOPANDAS_OK:
        return None
    if not (simd_csv.exists() and simd_zones.exists()):
        print("\n[SIMD] Files not found – skipping SIMD join")
        print(f"       Expected: {simd_csv}")
        print(f"                 {simd_zones}")
        print("       See SIMD_DOWNLOAD_INSTRUCTIONS at the top of this file.")
        return None

    print("\n[SIMD] Loading SIMD 2020 crime domain data...")
    try:
        ind = pd.read_csv(simd_csv)
        ind.columns = [c.lower().strip() for c in ind.columns]

        # Find datazone column
        dz_col = next(
            (c for c in ind.columns if c in ("data_zone", "datazone", "dz")), None
        )
        if dz_col is None:
            dz_col = next((c for c in ind.columns if "data_zone" in c), None)

        # Find crime column – prefer rate over count, over rank
        cr_col = next((c for c in ind.columns if "crime_rate" in c), None)
        if cr_col is None:
            cr_col = next((c for c in ind.columns if "crime_count" in c), None)
        if cr_col is None:
            cr_col = next((c for c in ind.columns if "crime" in c), None)

        if dz_col is None or cr_col is None:
            print(f"      [!] Cannot identify required columns in {simd_csv.name}")
            print(f"          Available: {list(ind.columns[:12])}")
            return None

        ind = (ind[[dz_col, cr_col]]
               .rename(columns={dz_col: "DataZone", cr_col: "crime_raw"})
               .copy())
        ind["crime_raw"] = pd.to_numeric(ind["crime_raw"], errors="coerce").fillna(0)

        # Normalise: clip at 95th percentile
        p95 = ind["crime_raw"].quantile(0.95) or 1.0
        ind["crime_norm"] = (ind["crime_raw"] / p95).clip(0, 1)

        zones = gpd.read_file(str(simd_zones))
        # Harmonise the join key (GeoJSON may call it 'DataZone' or 'datazone')
        zones.columns = [c if c == "geometry" else c for c in zones.columns]
        dz_col_geo = next(
            (c for c in zones.columns if c.lower().replace("_", "") == "datazone"), None
        )
        if dz_col_geo and dz_col_geo != "DataZone":
            zones = zones.rename(columns={dz_col_geo: "DataZone"})

        zones = zones.merge(ind[["DataZone", "crime_norm"]], on="DataZone", how="left")
        zones["crime_norm"] = zones["crime_norm"].fillna(0.5)
        zones = zones.set_crs("EPSG:4326", allow_override=True)

        print(f"      {len(zones):,} datazones loaded with SIMD crime scores")
        return zones

    except Exception as e:
        print(f"      [!] SIMD load failed: {e}")
        return None


def simd_batch_join(streets_df: pd.DataFrame, zones_gdf) -> dict[str, float]:
    """
    Vectorised spatial join: map each street midpoint to its datazone
    crime_norm score.  Returns dict: street_name → crime_norm.
    """
    print("      Running SIMD spatial join...")
    pts_gdf = gpd.GeoDataFrame(
        streets_df[["name", "mid_lat", "mid_lng"]].copy(),
        geometry=[Point(r.mid_lng, r.mid_lat) for _, r in streets_df.iterrows()],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        pts_gdf,
        zones_gdf[["geometry", "crime_norm"]],
        how="left",
        predicate="within",
    )
    # Streets near datazone boundaries may appear twice – keep first match
    joined = joined[~joined.index.duplicated(keep="first")]
    result = dict(zip(joined["name"], joined["crime_norm"].fillna(0.5)))
    matched = joined["crime_norm"].notna().sum()
    print(f"      {matched:,}/{len(streets_df):,} streets matched to datazones")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – Neighbourhood fallback helpers
# ══════════════════════════════════════════════════════════════════════════════

def _haversine_m(lat1, lng1, lat2, lng2) -> float:
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(p1) * math.cos(p2)
         * math.sin(math.radians(lng2 - lng1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _nearest_neighbourhood(lat: float, lng: float) -> tuple[float, float]:
    """Return (crime_rate, lighting_score) of the nearest neighbourhood."""
    best_d = float("inf")
    best   = (0.5, 0.5)
    for _, (nlat, nlng, crime, light) in NEIGHBOURHOOD_FALLBACK.items():
        d = _haversine_m(lat, lng, nlat, nlng)
        if d < best_d:
            best_d, best = d, (crime, light)
    return best


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 – Aggregate to one row per street and assign crime/lighting
# ══════════════════════════════════════════════════════════════════════════════

def build_street_table(
    segments: pd.DataFrame,
    police_norm: dict[str, float],
    simd_zones=None,
) -> pd.DataFrame:
    """
    Produce one row per unique named street:
      name, mid_lat, mid_lng, crime_rate_normalised, lighting_score

    Lighting: weighted average of per-segment scores (weighted by length).
    Crime:    Police API → SIMD spatial join → neighbourhood fallback.
    """
    print("\n[3/5] Computing per-segment lighting scores...")
    segments["lighting_score"] = segments.apply(compute_lighting, axis=1)

    print("[4/5] Aggregating to one row per street name...")

    def _agg(grp: pd.DataFrame) -> pd.Series:
        w = grp["length_m"].values
        if w.sum() == 0:
            w = np.ones(len(grp))
        return pd.Series({
            "mid_lat":       float(np.average(grp["mid_lat"],       weights=w)),
            "mid_lng":       float(np.average(grp["mid_lng"],       weights=w)),
            "lighting_score": float(np.average(grp["lighting_score"], weights=w)),
        })

    streets = segments.groupby("name").apply(_agg).reset_index()
    print(f"      {len(streets):,} unique streets")

    # ── SIMD batch join (if available) ─────────────────────────────────────────
    simd_crime: dict[str, float] = {}
    if simd_zones is not None:
        simd_crime = simd_batch_join(streets, simd_zones)

    # ── Assign crime rate ──────────────────────────────────────────────────────
    crime_rates  = []
    crime_source = []

    for _, row in streets.iterrows():
        name = row["name"]
        lat  = row["mid_lat"]
        lng  = row["mid_lng"]

        # 1. Police API  (street-name match)
        if name in police_norm:
            crime_rates.append(police_norm[name])
            crime_source.append("police_api")
            continue

        # 2. SIMD spatial join
        if name in simd_crime:
            crime_rates.append(simd_crime[name])
            crime_source.append("simd")
            continue

        # 3. Nearest neighbourhood heuristic
        crime, _ = _nearest_neighbourhood(lat, lng)
        crime_rates.append(crime)
        crime_source.append("neighbourhood")

    streets["crime_rate_normalised"] = crime_rates
    streets["crime_source"]          = crime_source

    src_counts = pd.Series(crime_source).value_counts()
    print("      Crime data source breakdown:")
    for src, cnt in src_counts.items():
        pct = 100 * cnt / len(streets)
        print(f"        {src:<22s} {cnt:4d}  ({pct:.0f}%)")

    streets["crime_rate_normalised"] = streets["crime_rate_normalised"].round(4)
    streets["lighting_score"]        = streets["lighting_score"].round(4)
    streets["mid_lat"]               = streets["mid_lat"].round(6)
    streets["mid_lng"]               = streets["mid_lng"].round(6)

    return streets.sort_values("name").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 – Save outputs
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(streets: pd.DataFrame) -> None:
    """
    Write two files:
      1. edinburgh_streets.csv  – full reference table (all columns)
      2. streets_list.py        – Python STREETS constant for seed_data.py
    """
    print("\n[5/5] Saving outputs...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    streets[["name", "mid_lat", "mid_lng",
             "crime_rate_normalised", "lighting_score", "crime_source"]
    ].to_csv(str(OUTPUT_CSV), index=False)
    print(f"      → {OUTPUT_CSV}  ({len(streets):,} rows)")

    # ── Python snippet ────────────────────────────────────────────────────────
    lines = [
        '"""',
        "Edinburgh streets – auto-generated by extract_streets.py",
        f"Generated : {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC",
        f"Streets   : {len(streets)}",
        '"""',
        "",
        "# Each entry: (name, latitude, longitude, crime_rate_normalised, lighting_score)",
        "STREETS = [",
    ]
    for _, row in streets.iterrows():
        name = str(row["name"]).replace('"', '\\"')
        lines.append(
            f'    ("{name}", {row["mid_lat"]:.6f}, {row["mid_lng"]:.6f}, '
            f'{row["crime_rate_normalised"]:.4f}, {row["lighting_score"]:.4f}),'
        )
    lines += ["]", ""]
    OUTPUT_PY.write_text("\n".join(lines), encoding="utf-8")
    print(f"      → {OUTPUT_PY}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Streets extracted   : {len(streets):,}")
    print(f"  Crime source mix    : {streets['crime_source'].value_counts().to_dict()}")
    print(f"  Avg crime rate      : {streets['crime_rate_normalised'].mean():.3f}")
    print(f"  Avg lighting score  : {streets['lighting_score'].mean():.3f}")
    print(f"  Lat range           : {streets['mid_lat'].min():.4f} – {streets['mid_lat'].max():.4f}")
    print(f"  Lng range           : {streets['mid_lng'].min():.4f} – {streets['mid_lng'].max():.4f}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review data/edinburgh_streets.csv to check data quality")
    print("  2. The STREETS list is ready in data/streets_list.py")
    print("  3. See the seed_data.py update instructions in that file")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  SafeRoutes – Edinburgh Street Data Extraction")
    print("=" * 60)

    # 1. Extract named segments from OSMnx
    segments = extract_segments(GRAPH_PATH, OSM_BBOX)

    # 2. Police Scotland crime data (free REST API, no key needed)
    raw_counts = fetch_police_crimes(OSM_BBOX, n_months=6)
    police_norm = normalise_crime_counts(raw_counts)

    # 3. SIMD spatial join (optional – only if files are present)
    simd_zones = load_simd(SIMD_CSV, SIMD_ZONES)

    # 4. Build final one-row-per-street table
    streets = build_street_table(segments, police_norm, simd_zones)

    # 5. Write outputs
    save_outputs(streets)


if __name__ == "__main__":
    main()
