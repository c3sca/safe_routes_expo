"""
seed_data.py – Populate the SafeRoutes database with realistic Edinburgh data.

Run once (or whenever the database is empty) to create:
  - 20 fictional university student users
  - 20 Edinburgh area records with crime/lighting baselines
  - ~100 safety ratings spread across areas and times of day

This seed data is the training set for all ML models.
"""
import random
from datetime import datetime, timedelta

# Edinburgh neighbourhoods: (name, lat, lng, crime_rate, lighting_score)
# crime_rate: 0=low crime, 1=high crime  (normalised from Police Scotland data)
# lighting_score: 0=dark streets, 1=well-lit
AREAS = [
    ("Old Town",       55.9486, -3.1999, 0.65, 0.70),
    ("New Town",       55.9558, -3.1962, 0.30, 0.85),
    ("Leith",          55.9756, -3.1724, 0.60, 0.60),
    ("The Meadows",    55.9400, -3.1890, 0.25, 0.55),
    ("Marchmont",      55.9360, -3.1870, 0.15, 0.65),
    ("Bruntsfield",    55.9360, -3.2020, 0.20, 0.70),
    ("Tollcross",      55.9432, -3.2025, 0.50, 0.75),
    ("Cowgate",        55.9477, -3.1890, 0.75, 0.50),
    ("Grassmarket",    55.9471, -3.1955, 0.70, 0.60),
    ("Lothian Road",   55.9460, -3.2055, 0.68, 0.80),
    ("Calton Hill",    55.9553, -3.1790, 0.45, 0.40),
    ("Haymarket",      55.9460, -3.2190, 0.35, 0.80),
    ("Morningside",    55.9270, -3.2050, 0.10, 0.75),
    ("Newington",      55.9360, -3.1840, 0.20, 0.70),
    ("Polwarth",       55.9340, -3.2180, 0.15, 0.65),
    ("Gorgie",         55.9350, -3.2400, 0.45, 0.60),
    ("Stockbridge",    55.9590, -3.2130, 0.12, 0.80),
    ("Dean Village",   55.9540, -3.2210, 0.10, 0.50),
    ("Fountainbridge", 55.9427, -3.2110, 0.40, 0.72),
    ("Lauriston",      55.9435, -3.1940, 0.35, 0.68),
]

# Fictional student users: (username, email, university)
USERS = [
    ("aisling_m",   "aisling.m@ed.ac.uk",      "University of Edinburgh"),
    ("zoe_r",       "zoe.r@ed.ac.uk",           "University of Edinburgh"),
    ("priya_k",     "priya.k@hw.ac.uk",         "Heriot-Watt University"),
    ("lucy_j",      "lucy.j@ed.ac.uk",          "University of Edinburgh"),
    ("mei_chen",    "mei.chen@ed.ac.uk",        "University of Edinburgh"),
    ("niamh_o",     "niamh.o@napier.ac.uk",     "Edinburgh Napier University"),
    ("sarah_b",     "sarah.b@ed.ac.uk",         "University of Edinburgh"),
    ("fatima_a",    "fatima.a@ed.ac.uk",        "University of Edinburgh"),
    ("claire_d",    "claire.d@caledonian.ac.uk","Edinburgh Caledonian University"),
    ("rosa_v",      "rosa.v@ed.ac.uk",          "University of Edinburgh"),
    ("elena_p",     "elena.p@hw.ac.uk",         "Heriot-Watt University"),
    ("sophie_t",    "sophie.t@ed.ac.uk",        "University of Edinburgh"),
    ("anna_w",      "anna.w@napier.ac.uk",      "Edinburgh Napier University"),
    ("jade_h",      "jade.h@ed.ac.uk",          "University of Edinburgh"),
    ("nadia_c",     "nadia.c@ed.ac.uk",         "University of Edinburgh"),
    ("emma_f",      "emma.f@hw.ac.uk",          "Heriot-Watt University"),
    ("tara_n",      "tara.n@ed.ac.uk",          "University of Edinburgh"),
    ("isla_m",      "isla.m@ed.ac.uk",          "University of Edinburgh"),
    ("chloe_s",     "chloe.s@napier.ac.uk",     "Edinburgh Napier University"),
    ("yuki_t",      "yuki.t@ed.ac.uk",          "University of Edinburgh"),
]

# Safety score profiles per area and time of day.
# Each entry: (area_name, time_of_day, mean_score, std_dev)
# mean_score is on the 1–5 scale — safer areas score higher.
RATING_PROFILES = [
    # Old Town – busy but some risk at night
    ("Old Town",       "day",     4.0, 0.6),
    ("Old Town",       "evening", 3.2, 0.7),
    ("Old Town",       "night",   2.5, 0.8),
    # New Town – generally very safe
    ("New Town",       "day",     4.5, 0.4),
    ("New Town",       "evening", 4.0, 0.5),
    ("New Town",       "night",   3.5, 0.6),
    # Leith – improving but still mixed
    ("Leith",          "day",     3.5, 0.7),
    ("Leith",          "evening", 2.8, 0.8),
    ("Leith",          "night",   2.2, 0.9),
    # The Meadows – safe by day, isolated at night
    ("The Meadows",    "day",     4.5, 0.4),
    ("The Meadows",    "evening", 3.8, 0.6),
    ("The Meadows",    "night",   2.8, 0.9),
    # Marchmont – quiet residential
    ("Marchmont",      "day",     4.7, 0.3),
    ("Marchmont",      "evening", 4.3, 0.4),
    ("Marchmont",      "night",   3.8, 0.5),
    # Bruntsfield – cafe district, safe
    ("Bruntsfield",    "day",     4.6, 0.3),
    ("Bruntsfield",    "evening", 4.2, 0.4),
    ("Bruntsfield",    "night",   3.5, 0.6),
    # Tollcross – busy junction, moderate
    ("Tollcross",      "day",     3.8, 0.5),
    ("Tollcross",      "evening", 3.2, 0.7),
    ("Tollcross",      "night",   2.6, 0.8),
    # Cowgate – nightlife heavy, lower scores especially at night
    ("Cowgate",        "day",     3.0, 0.8),
    ("Cowgate",        "evening", 2.5, 0.9),
    ("Cowgate",        "night",   1.8, 1.0),
    # Grassmarket – similar to Cowgate
    ("Grassmarket",    "day",     3.2, 0.7),
    ("Grassmarket",    "evening", 2.7, 0.8),
    ("Grassmarket",    "night",   2.0, 0.9),
    # Lothian Road – busy late-night strip
    ("Lothian Road",   "day",     3.5, 0.6),
    ("Lothian Road",   "evening", 2.8, 0.8),
    ("Lothian Road",   "night",   2.2, 1.0),
    # Calton Hill – scenic but isolated
    ("Calton Hill",    "day",     3.8, 0.6),
    ("Calton Hill",    "evening", 2.5, 0.9),
    ("Calton Hill",    "night",   1.8, 0.9),
    # Haymarket – busy transport hub
    ("Haymarket",      "day",     4.0, 0.5),
    ("Haymarket",      "evening", 3.5, 0.6),
    ("Haymarket",      "night",   2.8, 0.8),
    # Morningside – quiet and safe
    ("Morningside",    "day",     4.8, 0.2),
    ("Morningside",    "evening", 4.5, 0.3),
    ("Morningside",    "night",   4.0, 0.5),
    # Newington – student area, safe
    ("Newington",      "day",     4.3, 0.4),
    ("Newington",      "evening", 3.9, 0.5),
    ("Newington",      "night",   3.4, 0.6),
    # Polwarth – quiet residential
    ("Polwarth",       "day",     4.5, 0.3),
    ("Polwarth",       "evening", 4.0, 0.4),
    ("Polwarth",       "night",   3.5, 0.6),
    # Gorgie – more deprived, lower scores
    ("Gorgie",         "day",     3.2, 0.7),
    ("Gorgie",         "evening", 2.6, 0.8),
    ("Gorgie",         "night",   2.0, 0.9),
    # Stockbridge – village-like, very safe
    ("Stockbridge",    "day",     4.7, 0.3),
    ("Stockbridge",    "evening", 4.4, 0.4),
    ("Stockbridge",    "night",   3.9, 0.5),
    # Dean Village – beautiful but dark and isolated
    ("Dean Village",   "day",     4.3, 0.5),
    ("Dean Village",   "evening", 3.0, 0.8),
    ("Dean Village",   "night",   2.2, 0.9),
    # Fountainbridge – improving area
    ("Fountainbridge", "day",     3.8, 0.6),
    ("Fountainbridge", "evening", 3.2, 0.7),
    ("Fountainbridge", "night",   2.7, 0.8),
    # Lauriston – near uni, moderate
    ("Lauriston",      "day",     3.9, 0.5),
    ("Lauriston",      "evening", 3.4, 0.6),
    ("Lauriston",      "night",   2.8, 0.8),
]


def seed(app, db_instance):
    """
    Seed the database with realistic Edinburgh data.
    Safe to call multiple times – skips seeding if data already present.
    """
    from models.database import User, SafetyRating, Area

    with app.app_context():
        # Skip if already seeded
        if User.query.count() > 0:
            print("[seed] Database already contains data – skipping seed.")
            return

        random.seed(42)  # reproducible seed

        print("[seed] Seeding users...")
        user_objects = []
        for (uname, email, uni) in USERS:
            u = User(username=uname, email=email, university=uni)
            u.set_password("password123")  # default dev password
            db_instance.session.add(u)
            user_objects.append(u)
        db_instance.session.flush()  # get IDs without committing

        print("[seed] Seeding areas...")
        area_objects = {}
        for (name, lat, lng, crime, lighting) in AREAS:
            a = Area(
                name=name,
                latitude=lat,
                longitude=lng,
                crime_rate_normalised=crime,
                lighting_score=lighting,
                avg_user_score=0.5,        # will be updated after ratings
                composite_safety_score=0.5, # will be updated by ML pipeline
            )
            db_instance.session.add(a)
            area_objects[name] = a
        db_instance.session.flush()

        print("[seed] Seeding safety ratings...")
        # Build a lookup: area_name → Area object
        n_ratings = 0
        base_time = datetime.utcnow() - timedelta(days=180)

        for (area_name, time_of_day, mean_score, std_dev) in RATING_PROFILES:
            area = area_objects[area_name]

            # Generate 5–7 ratings per (area, time_of_day) combination
            n = random.randint(5, 7)
            for _ in range(n):
                # Sample a score from a Gaussian, then clamp to [1, 5]
                raw = random.gauss(mean_score, std_dev)
                score = max(1, min(5, round(raw)))

                # Pick a random user
                user = random.choice(user_objects)

                # Jitter the coordinates slightly so ratings aren't all identical
                lat = area.latitude  + random.uniform(-0.002, 0.002)
                lng = area.longitude + random.uniform(-0.002, 0.002)

                # Random timestamp within the last 6 months
                offset_days = random.randint(0, 180)
                created = base_time + timedelta(
                    days=offset_days,
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                )

                rating = SafetyRating(
                    user_id=user.id,
                    latitude=lat,
                    longitude=lng,
                    area_name=area_name,
                    safety_score=score,
                    time_of_day=time_of_day,
                    comment=None,
                    created_at=created,
                )
                db_instance.session.add(rating)
                n_ratings += 1

        # Compute initial avg_user_score for each area from the freshly seeded ratings
        db_instance.session.flush()
        for area in area_objects.values():
            ratings = SafetyRating.query.filter_by(area_name=area.name).all()
            if ratings:
                avg = sum(r.safety_score for r in ratings) / len(ratings)
                area.avg_user_score = (avg - 1) / 4   # normalise 1–5 → 0–1

        db_instance.session.commit()
        print(f"[seed] Done. {len(USERS)} users, {len(AREAS)} areas, {n_ratings} ratings.")
