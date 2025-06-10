# database_module/models.py
from sqlalchemy import Column, String, Date, Integer, Float, Text, JSON, DateTime
from datetime import datetime
from .db import Base

class ModelEntry(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    created = Column(DateTime, nullable=False, default=datetime.now)
    updated = Column(DateTime, nullable=True)  # Added updated field
    description = Column(Text, nullable=True)
    capabilities = Column(Text, nullable=True)  # Store model_capabilities

class DriftEntry(Base):
    __tablename__ = "drift_history"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False, index=True)
    date = Column(DateTime, nullable=False, default=datetime.now)
    drift_score = Column(Float, nullable=True)

class DiagnosticData(Base):
    __tablename__ = "diagnostic_data"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False, index=True)
    created = Column(DateTime, nullable=False, default=datetime.now)
    is_baseline = Column(Integer, nullable=False, default=0)  # 0=latest, 1=baseline
    questions = Column(JSON, nullable=True)
    answers = Column(JSON, nullable=True)