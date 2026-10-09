"""
Campus AI Backend — FastAPI Application Entry Point.

Initialises the ML engine on startup and mounts the API router with
CORS support for the React frontend.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure the backend package directory is on sys.path so that sibling
# modules (models, ml_engine, …) can be imported when running with
# ``uvicorn backend.main:app`` from the project root.
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from ml_engine import MLEngine  # noqa: E402
from router import router, set_engine  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("campus_ai")

# ---------------------------------------------------------------------------
# Lifespan — initialise the ML engine once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: load the dataset and fit the models.
    Shutdown: (nothing to clean up for now).
    """
    dataset_path = os.path.join(_BACKEND_DIR, "..", "dataset", "ML-Dataset.csv")
    dataset_path = os.path.normpath(dataset_path)

    logger.info("Loading ML engine from %s …", dataset_path)
    engine = MLEngine(dataset_path=dataset_path)
    set_engine(engine)
    logger.info("ML engine initialised successfully.")

    yield  # application is running

    logger.info("Shutting down Campus AI Backend.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Campus AI Backend",
    description=(
        "Real-time social-isolation detection API for university campuses. "
        "Uses K-Means clustering and Isolation Forest anomaly detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the API router
app.include_router(router)


# Root health check (convenience duplicate)
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint — quick health probe."""
    return {"service": "Campus AI Backend", "status": "running"}
