"""
API Router for the Campus AI Backend.

Exposes the ``/predict`` and ``/health`` endpoints.
"""

import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from ml_engine import MLEngine
from models import AssessmentInput, PredictionResponse
from privacy import append_training_data, strip_pii

logger = logging.getLogger("router")

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level reference to the ML engine — set by main.py on startup.
# ---------------------------------------------------------------------------
_engine: Optional[MLEngine] = None


def set_engine(engine: MLEngine) -> None:
    """Called by main.py to inject the initialised MLEngine instance."""
    global _engine
    _engine = engine


def get_engine() -> MLEngine:
    """Return the MLEngine or raise if not initialised."""
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="ML engine is not initialised yet. Please try again shortly.",
        )
    return _engine


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/predict", response_model=PredictionResponse)
async def predict(assessment: AssessmentInput) -> PredictionResponse:
    """
    Accept a student assessment and return a social-isolation prediction
    with cluster label, risk category, anomaly score, recommendation,
    and radar-chart data.
    """
    engine = get_engine()

    try:
        result = engine.predict(assessment)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {exc}",
        ) from exc

    # ---- Privacy-safe training-data logging ----
    try:
        raw_input = assessment.model_dump()
        clean_features = strip_pii(raw_input)
        prediction_data: Dict[str, Any] = {
            "social_density_score": result.social_density_score,
            "cluster_label": result.cluster_label,
            "risk_category": result.risk_category,
        }
        training_csv = os.path.join(
            os.path.dirname(__file__), "training_data.csv"
        )
        append_training_data(
            user_id=result.user_id,
            features=clean_features,
            prediction=prediction_data,
            filepath=training_csv,
        )
    except Exception:
        # Logging failure must never block the response
        logger.warning("Failed to append training data", exc_info=True)

    return result


@router.get("/health")
async def health() -> Dict[str, str]:
    """Lightweight health-check endpoint."""
    engine_status = "ready" if _engine is not None else "not_initialised"
    return {"status": "ok", "engine": engine_status}
