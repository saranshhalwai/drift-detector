# database_module/models.py
from sqlalchemy import Column, String, Date, Integer, Float, Text
from .db import Base

class ModelEntry(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    created = Column(Date, nullable=False)
    description = Column(Text, nullable=True)

class DriftEntry(Base):
    __tablename__ = "drift_history"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False, index=True)
    date = Column(Date, nullable=False)
    drift_score = Column(Float, nullable=False)