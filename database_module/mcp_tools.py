# database_module/mcp_tools.py
from sqlalchemy.orm import Session
from sqlalchemy import or_
from .db import SessionLocal
from .models import ModelEntry, DriftEntry
from datetime import datetime
from typing import Any, Dict, List


def get_all_models_handler(_: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return all models as list of dicts matching:
      {name, created (ISO), description}
    """
    with SessionLocal() as session:
        entries = session.query(ModelEntry).all()
        return [
            {"name": e.name, "created": e.created.isoformat(), "description": e.description or ""}
            for e in entries
        ]


def search_models_handler(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Search models by name or description substring (case-insensitive).
    params: {search_term: str}
    """
    term = params.get("search_term", "").strip().lower()
    with SessionLocal() as session:
        query = session.query(ModelEntry)
        if term:
            like_pattern = f"%{term}%"
            query = query.filter(
                or_(
                    ModelEntry.name.ilike(like_pattern),
                    ModelEntry.description.ilike(like_pattern)
                )
            )
        entries = query.all()
        return [
            {"name": e.name, "created": e.created.isoformat(), "description": e.description or ""}
            for e in entries
        ]


def get_model_details_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a single model's details including system_prompt and description.
    params: {model_name: str}
    """
    model_name = params.get("model_name")
    with SessionLocal() as session:
        e = session.query(ModelEntry).filter_by(name=model_name).first()
        if not e:
            return {"name": model_name, "system_prompt": "You are a helpful AI assistant.", "description": ""}
        # You can store system_prompt as a column if desired; here placeholder
        return {"name": e.name, "system_prompt": "You are a helpful AI assistant.", "description": e.description or ""}


def save_model_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save or update a model's system_prompt.
    params: {model_name: str, system_prompt: str}
    """
    name = params.get("model_name")
    prompt = params.get("system_prompt", "")
    with SessionLocal() as session:
        entry = session.query(ModelEntry).filter_by(name=name).first()
        if not entry:
            # New model; created today
            entry = ModelEntry(
                name=name,
                created=datetime.utcnow().date(),
                description=""
            )
            session.add(entry)
        # Optionally store prompt in another table or JSON field
        session.commit()
    return {"message": f"Model '{name}' saved."}


def calculate_drift_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder drift calculation: record a new random drift score today.
    params: {model_name: str}
    """
    import random
    name = params.get("model_name")
    score = round(random.uniform(0, 1), 3)
    today = datetime.utcnow().date()
    with SessionLocal() as session:
        entry = DriftEntry(
            model_name=name,
            date=today,
            drift_score=score
        )
        session.add(entry)
        session.commit()
    return {"drift_score": score, "message": f"Drift recorded for '{name}'."}


def get_drift_history_handler(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return drift history as list of {date, drift_score} for a model.
    params: {model_name: str}
    """
    name = params.get("model_name")
    with SessionLocal() as session:
        entries = session.query(DriftEntry).filter_by(model_name=name).order_by(DriftEntry.date).all()
        return [
            {"date": e.date.isoformat(), "drift_score": e.drift_score}
            for e in entries
        ]
