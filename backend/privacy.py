"""
Privacy module for the Campus AI Backend.

Provides session-ID generation, PII stripping, and thread-safe
training-data logging — all without retaining personally identifiable
information.
"""

import os
import uuid
import threading
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd

# Module-level lock for thread-safe CSV appending
_csv_lock = threading.Lock()

# Columns that are written to the training-data CSV
_TRAINING_COLUMNS = [
    "user_id",
    "friend_group_size",
    "friend_influence",
    "cgpa",
    "class_attendance",
    "daily_self_study_hours",
    "society_memberships",
    "society_events_attended",
    "meals_with_friends",
    "hours_alone_weekly",
    "common_room_frequency",
    "social_outings",
    "assignments_due",
    "workload_stress",
    "workload_management",
    "screen_time",
    "sleep_hours",
    "social_wellness_rating",
    "mood_rating",
    "loneliness_score",
    "peer_satisfaction",
    "peer_helpfulness",
    "social_density_score",
    "cluster_label",
    "risk_category",
    "timestamp_utc",
]

# Fields that may carry PII — stripped before any downstream use
_PII_FIELDS = {"name", "timestamp", "email", "phone", "free_text", "comments", "notes"}


def generate_session_id() -> str:
    """Return a 12-character hex session token (no PII)."""
    return uuid.uuid4().hex[:12]


def strip_pii(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove any personally-identifiable or free-text fields from *data*
    and return only the numerical/categorical features needed for
    inference.

    Parameters
    ----------
    data : dict
        Raw input dictionary (may include name, timestamp, etc.).

    Returns
    -------
    dict
        Cleaned dictionary with only numerical feature values.
    """
    cleaned: Dict[str, Any] = {}
    for key, value in data.items():
        lower_key = key.lower()
        # Skip known PII fields
        if lower_key in _PII_FIELDS:
            continue
        # Skip non-numeric string values that aren't the living_status flag
        if isinstance(value, str) and lower_key != "living_status":
            continue
        cleaned[key] = value
    return cleaned


def append_training_data(
    user_id: str,
    features: Dict[str, Any],
    prediction: Dict[str, Any],
    filepath: str = "training_data.csv",
) -> None:
    """
    Append a single prediction record to the training-data CSV in a
    thread-safe manner.

    If the file does not yet exist it is created with the appropriate
    header row.

    Parameters
    ----------
    user_id : str
        Anonymous session identifier.
    features : dict
        Numerical feature values used for the prediction.
    prediction : dict
        Model output containing social_density_score, cluster_label,
        and risk_category.
    filepath : str
        Path to the CSV file (created if missing).
    """
    row: Dict[str, Any] = {"user_id": user_id}

    # Merge numerical features
    for col in _TRAINING_COLUMNS:
        if col == "user_id":
            continue
        if col == "timestamp_utc":
            continue
        if col in features:
            row[col] = features[col]
        elif col in prediction:
            row[col] = prediction[col]
        else:
            row[col] = None

    row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    with _csv_lock:
        file_exists = os.path.isfile(filepath)
        df_row = pd.DataFrame([row], columns=_TRAINING_COLUMNS)
        df_row.to_csv(
            filepath,
            mode="a",
            header=not file_exists,
            index=False,
        )
