"""
SQLAlchemy ORM models for SafeRoutes.

Tables:
  users          – registered users (university students)
  safety_ratings – crowd-sourced safety ratings submitted by users
  areas          – Edinburgh neighbourhood records with ML-derived scores
  routes         – saved routes generated for users
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """Registered user of SafeRoutes."""
    __tablename__ = "users"

    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(64),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    university    = db.Column(db.String(128), default="University of Edinburgh")
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)

    # Relationship back to ratings
    ratings = db.relationship("SafetyRating", backref="user", lazy=True)
    routes  = db.relationship("Route",         backref="user", lazy=True)

    def set_password(self, password: str) -> None:
        """Hash and store a plaintext password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Return True if the given password matches the stored hash."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self) -> str:
        return f"<User {self.username}>"


class SafetyRating(db.Model):
    """
    A single crowd-sourced safety rating submitted by a user for a location.

    safety_score: integer 1–5 (1 = very unsafe, 5 = very safe)
    time_of_day:  'day', 'evening', or 'night'
    """
    __tablename__ = "safety_ratings"

    id          = db.Column(db.Integer,  primary_key=True)
    user_id     = db.Column(db.Integer,  db.ForeignKey("users.id"), nullable=False)
    latitude    = db.Column(db.Float,    nullable=False)
    longitude   = db.Column(db.Float,    nullable=False)
    area_name   = db.Column(db.String(128))
    safety_score = db.Column(db.Integer, nullable=False)   # 1–5
    time_of_day = db.Column(db.String(16), default="day")  # day/evening/night
    comment     = db.Column(db.Text,     nullable=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<SafetyRating user={self.user_id} area={self.area_name} score={self.safety_score}>"


class Area(db.Model):
    """
    An Edinburgh neighbourhood with aggregated safety scores.

    avg_user_score         – simple mean of all user ratings (0–1 normalised)
    crime_rate_normalised  – 0 = no crime, 1 = highest crime
    lighting_score         – 0 = dark, 1 = well-lit
    composite_safety_score – ML-derived weighted score (0–1, higher = safer)
    """
    __tablename__ = "areas"

    id                     = db.Column(db.Integer, primary_key=True)
    name                   = db.Column(db.String(128), unique=True, nullable=False)
    latitude               = db.Column(db.Float,  nullable=False)
    longitude              = db.Column(db.Float,  nullable=False)
    avg_user_score         = db.Column(db.Float,  default=0.5)
    crime_rate_normalised  = db.Column(db.Float,  default=0.5)
    lighting_score         = db.Column(db.Float,  default=0.5)
    composite_safety_score = db.Column(db.Float,  default=0.5)
    last_updated           = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Area {self.name} composite={self.composite_safety_score:.2f}>"


class Route(db.Model):
    """A safety-optimised walking route generated for a user."""
    __tablename__ = "routes"

    id              = db.Column(db.Integer,   primary_key=True)
    user_id         = db.Column(db.Integer,   db.ForeignKey("users.id"), nullable=True)
    start_lat       = db.Column(db.Float,     nullable=False)
    start_lng       = db.Column(db.Float,     nullable=False)
    end_lat         = db.Column(db.Float,     nullable=False)
    end_lng         = db.Column(db.Float,     nullable=False)
    route_geometry  = db.Column(db.Text,      nullable=True)   # JSON list of [lat,lng]
    safety_score    = db.Column(db.Float,     nullable=True)
    distance_km     = db.Column(db.Float,     nullable=True)
    algorithm       = db.Column(db.String(16), default="astar") # 'astar' or 'dijkstra'
    created_at      = db.Column(db.DateTime,  default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Route id={self.id} safety={self.safety_score:.2f} dist={self.distance_km:.2f}km>"
