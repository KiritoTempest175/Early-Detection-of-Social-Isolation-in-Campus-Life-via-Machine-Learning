"""Independent, leakage-aware comparisons for three unrelated student surveys.

Research only; do not combine labels for stress type, depression, and loneliness.
No raw respondent rows, names or individual predictions are exported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from ml_engine import COLUMN_MAP  # exact aliases shared with the dashboard

SEED = 42
MODEL_NAMES = ("Logistic Regression", "Random Forest", "Gradient Boosting", "SVM", "KNN")
SOCIAL_FEATURES = (
    "friend_group_size", "friend_influence", "cgpa", "class_attendance",
    "daily_self_study_hours", "society_memberships", "society_events_attended",
    "meals_with_friends", "hours_alone_weekly", "common_room_frequency",
    "social_outings", "assignments_due", "workload_stress",
    "workload_management", "screen_time", "sleep_hours",
    "social_wellness_rating", "mood_rating", "peer_satisfaction",
    "peer_helpfulness",
)
# Self-reported loneliness is the target: NEVER a predictor.
def load_campus(path: Path):
    df = pd.read_csv(path).rename(columns=COLUMN_MAP)
    missing = set(SOCIAL_FEATURES + ("loneliness_score",)) - set(df.columns)
    if missing:
        raise ValueError("Campus survey missing: " + ", ".join(sorted(missing)))
    data = df.loc[:, list(SOCIAL_FEATURES) + ["loneliness_score"]].copy()
    for col in SOCIAL_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    # The original questionnaire contains invalid outliers (e.g. 100000 quizzes).
    bounds = {"cgpa": (0, 4), "screen_time": (0, 24),
              "assignments_due": (0, 40), "daily_self_study_hours": (0, 24),
              "sleep_hours": (0, 24), "hours_alone_weekly": (0, 168)}
    for col, (minimum, maximum) in bounds.items():
        data[col] = data[col].clip(minimum, maximum)
    target = pd.to_numeric(data.pop("loneliness_score"), errors="coerce")
    valid = target.notna() & target.between(1, 5)
    return data.loc[valid].reset_index(drop=True), (target.loc[valid] >= 4).astype(int).reset_index(drop=True)

def load_stress(path: Path):
    df = pd.read_csv(path)
    target_col = "Which type of stress do you primarily experience?"
    if target_col not in df:
        raise ValueError("Stress survey missing target column")
    description = df[target_col].fillna("").astype(str)
    # Label text varies; explicit Eustress vs Distress only.
    positive = description.str.contains("Distress", case=False, regex=False)
    negative = description.str.contains("Eustress", case=False, regex=False)
    valid = positive ^ negative
    features = df.drop(columns=[target_col]).loc[valid].reset_index(drop=True)
    # No respondent identifying information; numeric questionnaire values only.
    for col in features:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    y = positive.loc[valid].astype(int).reset_index(drop=True)
    return features, y

def load_mental_health(path: Path):
    df = pd.read_csv(path)
    target_col = "Do you have Depression?"
    cols = [
        "Age", "Your current year of Study", "What is your CGPA?",
        "Do you have Anxiety?", "Do you have Panic attack?",
    ]
    missing = set(cols + [target_col]) - set(df)
    if missing:
        raise ValueError("Mental-health survey missing: " + ", ".join(sorted(missing)))
    target = df[target_col].astype("string").str.strip().str.lower()
    valid = target.isin(["yes", "no"])
    features = df.loc[valid, cols].copy().reset_index(drop=True)
    features["Age"] = pd.to_numeric(features["Age"], errors="coerce")
    # Treatment-seeking and demographic identifiers deliberately excluded.
    return features, target.loc[valid].eq("yes").astype(int).reset_index(drop=True)

def model_factory():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=180, class_weight="balanced", random_state=SEED, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "SVM": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

def evaluate(name, X, y, output: Path):
    if len(y) < 30 or y.nunique() < 2 or y.value_counts().min() < 5:
        raise ValueError(name + ": too few observations per class for a holdout")
    # Drop identical records before splitting to mitigate duplicate-row leakage.
    frame = X.copy()
    frame["__target__"] = y.to_numpy()
    frame = frame.drop_duplicates().reset_index(drop=True)
    y = frame.pop("__target__").astype(int)
    X = frame
    if y.nunique() < 2 or y.value_counts().min() < 5:
        raise ValueError(name + ": too few unique observations per class")
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X if c not in numeric]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)
    ], sparse_threshold=0)
    results = {}
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(18, 3.8))
    for ax, (model_name, estimator) in zip(axes, model_factory().items()):
        pipeline = Pipeline([("preprocess", prep), ("classifier", estimator)])
        pipeline.fit(train_x, train_y)
        pred = pipeline.predict(test_x)
        prob = pipeline.predict_proba(test_x)[:, 1]
        matrix = confusion_matrix(test_y, pred, labels=[0, 1])
        metrics = {
            "accuracy": round(float(accuracy_score(test_y, pred)), 4),
            "precision_positive": round(float(precision_score(test_y, pred, zero_division=0)), 4),
            "recall_positive": round(float(recall_score(test_y, pred, zero_division=0)), 4),
            "f1_positive": round(float(f1_score(test_y, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(test_y, prob)), 4),
            "average_precision": round(float(average_precision_score(test_y, prob)), 4),
            "confusion_matrix": matrix.tolist(),
        }
        results[model_name] = metrics
        ax.imshow(matrix, cmap="Greys", vmin=0)
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(j, i, str(count), ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
        ax.set_title(model_name, fontsize=10)
    fig.suptitle(name + " | held-out confusion matrices", fontsize=13)
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    report = {
        "dataset": name, "unique_rows": len(y), "train_rows": len(train_y),
        "test_rows": len(test_y), "positive_test_count": int(test_y.sum()),
        "model_metrics": results,
        "caveats": [
            "Exploratory, one fixed random holdout; not external validation.",
            "Cross-sectional self-report, no future isolation prediction.",
            "Do not compare scores across surveys with different labels.",
            "No model trained on one survey is transferable to a different survey without validation.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "independent_comparisons")
    args = parser.parse_args()
    sources = [
        ("campus_loneliness_self_report", "ML-Dataset.csv", load_campus),
        ("stress_type_self_report", "Stress Dataset.csv", load_stress),
        ("depression_self_report", "Student Mental health.csv", load_mental_health),
    ]
    for name, filename, loader in sources:
        path = args.data_dir / filename
        if not path.is_file():
            raise SystemExit("Missing private dataset: " + str(path))
        X, y = loader(path)
        result = evaluate(name, X, y, args.output_dir / name)
        print(name + ": " + str(result["unique_rows"]) + " unique records")
        for model, metrics in result["model_metrics"].items():
            print("  " + model + ": F1=" + str(metrics["f1_positive"]) + ", AUC=" + str(metrics["roc_auc"]))
    print("Independent reports written to " + str(args.output_dir))
    print("These are unrelated outcomes. Do not compare them as a single prediction task.")

if __name__ == "__main__":
    main()
