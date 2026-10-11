"""Public, aggregate-only API. Assessment responses are never saved."""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from ml_engine import MLEngine
from models import AssessmentInput, PredictionResponse

router = APIRouter()
logger = logging.getLogger("campus.api")
_engine: Optional[MLEngine] = None

def set_engine(engine: Optional[MLEngine]) -> None:
    global _engine
    _engine = engine

def get_engine() -> MLEngine:
    if _engine is None:
        raise HTTPException(503, "Model unavailable. Configure DATASET_PATH or explicitly enable DEMO_MODE.")
    return _engine

@router.get("/health")
def health():
    return {"status": "ready" if _engine else "unavailable",
            "mode": ("synthetic_demo" if _engine.demo_mode else "survey") if _engine else "unavailable"}

@router.get("/api/overview")
def overview():
    return get_engine().overview()

@router.post("/predict", response_model=PredictionResponse)
def predict(assessment: AssessmentInput):
    try:
        return get_engine().predict(assessment)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(500, "Prediction could not be completed") from None
