# database_module/mcp_tools.py
from sqlalchemy.orm import Session
from sqlalchemy import or_
from .db import SessionLocal
from .models import ModelEntry, DriftEntry, DiagnosticData
from datetime import datetime
from typing import Any, Dict, List, Optional
import json


def get_all_models_handler(_: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return all models as list of dicts matching:
      {name, created (ISO), description}
    """
    with SessionLocal() as session:
        entries = session.query(ModelEntry).all()
        return [
            {
                "name": e.name,
                "created": e.created.isoformat() if e.created else datetime.now().isoformat(),
                "description": e.description or ""
            }
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
            {
                "name": e.name,
                "created": e.created.isoformat() if e.created else datetime.now().isoformat(),
                "description": e.description or ""
            }
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
            return {
                "name": model_name,
                "system_prompt": "You are a helpful AI assistant.",
                "description": ""
            }

        # Extract system prompt from capabilities if available
        system_prompt = "You are a helpful AI assistant."
        if e.capabilities and "System Prompt: " in e.capabilities:
            system_prompt = e.capabilities.split("System Prompt: ")[1]

        return {
            "name": e.name,
            "system_prompt": system_prompt,
            "description": e.description or ""
        }


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
                created=datetime.now(),
                description="",
                capabilities=f"System Prompt: {prompt}"
            )
            session.add(entry)
        else:
            # Update existing model
            entry.capabilities = f"System Prompt: {prompt}"
            entry.updated = datetime.now()

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
    today = datetime.now()
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


# === New functions for drift detection database operations ===

def save_diagnostic_data(
        model_name: str,
        questions: list,
        answers: list,
        is_baseline: bool = False
) -> None:
    """
    Save diagnostic questions and answers to the database
    """
    with SessionLocal() as session:
        # Check if model exists, create if not
        model = session.query(ModelEntry).filter_by(name=model_name).first()
        if not model:
            model = ModelEntry(
                name=model_name,
                created=datetime.now(),
                description=""
            )
            session.add(model)

        # Create new diagnostic entry
        diagnostic = DiagnosticData(
            model_name=model_name,
            is_baseline=1 if is_baseline else 0,
            questions=questions,
            answers=answers,
            created=datetime.now()
        )
        session.add(diagnostic)
        session.commit()


def get_baseline_diagnostics(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve baseline diagnostics for a model
    """
    with SessionLocal() as session:
        baseline = session.query(DiagnosticData) \
            .filter_by(model_name=model_name, is_baseline=1) \
            .order_by(DiagnosticData.created.desc()) \
            .first()

        if not baseline:
            return None

        return {
            "questions": baseline.questions,
            "answers": baseline.answers,
            "created": baseline.created.isoformat()
        }


def save_drift_score(model_name: str, drift_score: str) -> None:
    """
    Save drift score to database
    """
    # Try to convert score to float if possible
    try:
        score_float = float(drift_score)
    except ValueError:
        score_float = None

    with SessionLocal() as session:
        entry = DriftEntry(
            model_name=model_name,
            date=datetime.now(),
            drift_score=score_float
        )
        session.add(entry)
        session.commit()


def register_model_with_capabilities(model_name: str, capabilities: str) -> None:
    """
    Register a model with capabilities or update if already exists
    """
    with SessionLocal() as session:
        model = session.query(ModelEntry).filter_by(name=model_name).first()

        if model:
            model.capabilities = capabilities
            model.updated = datetime.now()
        else:
            model = ModelEntry(
                name=model_name,
                created=datetime.now(),
                capabilities=capabilities,
                description=""
            )
            session.add(model)

        session.commit()