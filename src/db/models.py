"""
Database models for NeendAI
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from .database import Base


class User(Base):
    """User model for storing user profiles"""

    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255))
    emergency_contact = Column(String(20))
    clinician_email = Column(String(255))
    device_token = Column(String(255))  # For push notifications
    alert_threshold = Column(Integer, default=30)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    sessions = relationship("SleepSession", back_populates="user", cascade="all, delete-orphan")


class SleepSession(Base):
    """Sleep session model for storing recording sessions"""

    __tablename__ = "sleep_sessions"

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    duration_hours = Column(Float)
    ahi_score = Column(Float)
    severity = Column(String(20))
    apnea_events = Column(Integer, default=0)
    hypopnea_events = Column(Integer, default=0)
    snoring_events = Column(Integer, default=0)
    total_events = Column(Integer, default=0)
    audio_file_url = Column(Text)  # S3 URL
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="sessions")
    events = relationship("SleepEvent", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint('duration_hours >= 0', name='check_duration_positive'),
        CheckConstraint('ahi_score >= 0', name='check_ahi_positive'),
    )


class SleepEvent(Base):
    """Individual sleep event model"""

    __tablename__ = "sleep_events"

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sleep_sessions.session_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    event_type = Column(
        String(20),
        CheckConstraint("event_type IN ('normal', 'snoring', 'hypopnea', 'apnea')"),
        nullable=False
    )
    confidence = Column(Float)
    duration_seconds = Column(Integer)
    audio_segment_url = Column(Text)
    reasoning = Column(Text)  # LLM reasoning for classification
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("SleepSession", back_populates="events")

    __table_args__ = (
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_confidence_range'),
    )
