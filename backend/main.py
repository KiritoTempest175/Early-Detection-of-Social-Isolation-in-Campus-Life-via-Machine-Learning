"""Minimal, privacy-first exploratory campus survey API."""
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml_engine import MLEngine
from router import router, set_engine, get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("campus")
DATASET_PATH = os.getenv("DATASET_PATH", str(Path(__file__).resolve().parent.parent / "dataset" / "ML-Dataset.csv"))
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
if "*" in ALLOWED_ORIGINS:
    raise RuntimeError("Wildcard CORS origins are not supported")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        engine = MLEngine(dataset_path=DATASET_PATH, demo_mode=DEMO_MODE)
        set_engine(engine)
        logger.info("Model ready, mode=%s", "synthetic_demo" if DEMO_MODE else "survey")
    except Exception:
        set_engine(None)
        logger.exception("Model unavailable; /predict returns 503 until configured")
    yield
    set_engine(None)

app = FastAPI(
    title="Campus Signals API", version="2.0.0",
    description="Exploratory campus survey analytics, not a diagnosis or validated clinical screening instrument.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"],
)
_lock = threading.Lock()
_requests = {}
@app.middleware("http")
async def security(request: Request, call_next):
    if request.url.path == "/predict" and request.method == "POST":
        raw_length = request.headers.get("content-length")
        if raw_length:
            try:
                if int(raw_length) > 8192:
                    return JSONResponse({"detail": "Request too large"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Invalid content length"}, status_code=400)
        key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with _lock:
            if len(_requests) > 5000:
                _requests.clear()
            remaining = [t for t in _requests.get(key, []) if now - t < 60]
            if len(remaining) >= 30:
                return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
            remaining.append(now)
            _requests[key] = remaining
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response

app.include_router(router)

@app.get("/")
def root():
    return {"service": "Campus Signals", "status": "online", "note": "Exploratory research, not clinical advice"}
