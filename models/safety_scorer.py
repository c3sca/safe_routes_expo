"""
safety_scorer.py – Composite safety score calculation utilities.

This module provides helper functions for:
  • Computing/updating composite scores for areas on demand
  • Generating heatmap data for the frontend
  • Looking up the nearest area record for a given lat/lng
"""
import math
import numpy as np
from typing import Optional, List, Dict


def haversine_dist(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return distance in metres between two (lat, lng) points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lng2 - lng1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def find_nearest_area(lat: float, lng: float, areas) -> Optional[object]:
    """
    Return the Area ORM object closest to (lat, lng).

    Parameters
    ----------
    areas : list of Area ORM objects
    """
    if not areas:
        return None
    closest = min(areas, key=lambda a: haversine_dist(lat, lng, a.latitude, a.longitude))
    return closest


def compute_composite(bayesian_score: float,
                      crime_rate: float,
                      lighting_score: float,
                      w1: float = 0.5,
                      w2: float = 0.3,
                      w3: float = 0.2) -> float:
    """
    Combine three safety signals into one composite score.

    Formula:
      composite = w1 * bayesian_score
                + w2 * (1 - crime_rate)
                + w3 * lighting_score

    All inputs and the output are in [0, 1] where 1 = safest.

    Parameters
    ----------
    bayesian_score : float – posterior mean from Bayesian estimator
    crime_rate     : float – normalised crime rate (0=safe, 1=dangerous)
    lighting_score : float – lighting quality (0=dark, 1=well-lit)
    w1, w2, w3     : float – weights (must sum to 1.0)
    """
    composite = (w1 * bayesian_score
                 + w2 * (1.0 - crime_rate)
                 + w3 * lighting_score)
    return float(np.clip(composite, 0.0, 1.0))


def update_area_score(area, db_session,
                      bayes_estimator=None,
                      w1: float = 0.5,
                      w2: float = 0.3,
                      w3: float = 0.2) -> float:
    """
    Recompute and persist the composite safety score for one area.

    Called after a new rating is submitted so the area score stays current.

    Parameters
    ----------
    area          : Area ORM object
    db_session    : SQLAlchemy db.session
    bayes_estimator : BayesianSafetyEstimator instance (optional)

    Returns
    -------
    float : updated composite_safety_score
    """
    from models.database import SafetyRating

    # Get all ratings for this area
    ratings = SafetyRating.query.filter_by(area_name=area.name).all()

    if not ratings:
        return area.composite_safety_score

    # Simple average (normalised)
    avg_raw = sum(r.safety_score for r in ratings) / len(ratings)
    avg_norm = (avg_raw - 1) / 4.0       # 1–5 → 0–1
    area.avg_user_score = avg_norm

    # Use Bayesian estimator if available, otherwise use simple average
    if bayes_estimator is not None:
        norm_scores = [(r.safety_score - 1) / 4.0 for r in ratings]
        b_score = bayes_estimator.estimate(norm_scores)
    else:
        b_score = avg_norm

    new_composite = compute_composite(
        b_score, area.crime_rate_normalised, area.lighting_score,
        w1, w2, w3
    )
    area.composite_safety_score = new_composite
    db_session.commit()
    return new_composite


def get_heatmap_data(areas, filter_mode: str = "composite") -> List[Dict]:
    """
    Prepare heatmap point data for Leaflet.heat.

    Parameters
    ----------
    areas       : list of Area ORM objects
    filter_mode : 'composite' | 'user' | 'crime'

    Returns
    -------
    list of dicts: [{"lat": ..., "lng": ..., "score": ...}, ...]
    where score is in [0, 1] with 1 = safest.
    """
    result = []
    for area in areas:
        if filter_mode == "user":
            score = area.avg_user_score
        elif filter_mode == "crime":
            score = 1.0 - area.crime_rate_normalised   # invert: low crime = safe
        else:
            score = area.composite_safety_score

        result.append({
            "lat":   area.latitude,
            "lng":   area.longitude,
            "name":  area.name,
            "score": round(float(score), 3),
        })
    return result
