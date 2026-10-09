"""
================================================================================
Multi-Dataset ML Pipeline with Comprehensive Accuracy Comparison
================================================================================
Trains 5 supervised classifiers on 2 dataset sets and produces side-by-side
accuracy comparison with full evaluation metrics.

  Set 1: ML-Dataset.csv (Social Isolation Detection)
  Set 2: Stress Dataset.csv + Student Mental health.csv (Stress & Mental Health)

Models:
  1. Random Forest
  2. Logistic Regression
  3. Support Vector Machine (SVM)
  4. K-Nearest Neighbors (KNN)
  5. Gradient Boosting

Authors: Muhammad Mobeen, Huzaifa Zaman
Course: Machine Learning (NUTECH, BSAI-2024)
Instructor: Lec Faiza Khan
================================================================================
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RANDOM_SEED = 42

# Resolve project root (parent of src/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs")
DATASET_DIR = os.path.join(_PROJECT_ROOT, "dataset")

# Column renaming map for ML-Dataset.csv (from existing pipeline)
COLUMN_MAP = {
    "Timestamp": "timestamp",
    "Name:": "name",
    "University/Institution Name \nWrite abbreviation only (e.g NUST, NUTECH, FAST)": "university",
    "Semester:": "semester",
    "Living Status": "living_status",
    "How many friends are in your group?": "friend_group_size",
    "How much does the presence of friends influence your decision to attend university events or classes?  ": "friend_influence",
    "Academic Performance:": "academic_performance",
    "Current CGPA (e.g., 3.45)": "cgpa",
    "Average Class Attendance Rate (Weekly)": "class_attendance",
    "Average Daily Self-Study Hours": "daily_self_study_hours",
    "Number of Active Society Memberships": "society_memberships",
    "Do you contribute/help/join society events?": "contributes_to_societies",
    "How many society meetings/events did you attend this semester?": "society_events_attended",
    "Do you play for any sports at University?": "plays_sports",
    "Do you enjoy physical Campus Presence?": "enjoys_campus_presence",
    "Number of meals eaten with friends on average in a week?": "meals_with_friends",
    "Total hours spent in personal space/alone on average in a week?": "hours_alone_weekly",
    "Frequency of using Common Rooms/Student Lounges/Library Rooms": "common_room_frequency",
    "How many times did you leave campus for social reasons this week? (Gatherings, events, trips, hangout, etc)": "social_outings",
    "Number of Assignments/Quizzes due this week?": "assignments_due",
    "Perceived Academic Workload Stress": "workload_stress",
    "How well did you manage your workload this week?": "workload_management",
    "Average daily Screen Time on your phone (Hours)? (In weekdays)": "screen_time",
    "Average hours of sleep per night? (in weekdays)": "sleep_hours",
    "Do you mostly skip breakfast/lunch in the cafeteria?": "skips_meals",
    "How would you rate your social interaction and wellness?": "social_wellness_rating",
    "How would you rate your overall mood this week?": "mood_rating",
    "How often did you feel socially isolated or lonely this week?": "loneliness_score",
    "Satisfaction with your peer interactions": "peer_satisfaction",
    "Which of the following spaces do you primarily use for studying/academic work? (Select all that apply)": "study_spaces",
    "Rank the following factors based on how much they contribute to your feeling of social isolation (1 being the highest contributor): [Heavy academic workload]": "isolation_factor_workload",
    "Rank the following factors based on how much they contribute to your feeling of social isolation (1 being the highest contributor): [Lack of common interest with peers]": "isolation_factor_peers",
    "Rank the following factors based on how much they contribute to your feeling of social isolation (1 being the highest contributor): [Digital distractions (Screen time)]": "isolation_factor_digital",
    "Rank the following factors based on how much they contribute to your feeling of social isolation (1 being the highest contributor): [Living status (Hostelite vs. Day Scholar)]": "isolation_factor_living",
    "How would you rate the helpfulness of your peers/friends in academic matters (e.g., studying for exams, assignments)?": "peer_helpfulness",
}

# Behavioral features for Set 1 modeling
SET1_FEATURES = [
    "friend_group_size", "friend_influence", "cgpa", "class_attendance",
    "daily_self_study_hours", "society_memberships", "society_events_attended",
    "meals_with_friends", "hours_alone_weekly", "common_room_frequency",
    "social_outings", "assignments_due", "workload_stress", "workload_management",
    "screen_time", "sleep_hours", "social_wellness_rating", "mood_rating",
    "peer_satisfaction", "peer_helpfulness",
]


# ==============================================================================
# MODEL FACTORY
# ==============================================================================
def get_models():
    """Return dictionary of model name -> configured model instance."""
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, class_weight='balanced',
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, class_weight='balanced',
        ),
        'SVM': SVC(
            kernel='rbf', probability=True, random_state=RANDOM_SEED,
            class_weight='balanced',
        ),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_SEED,
        ),
    }


# ==============================================================================
# SET 1: ML-Dataset.csv Preprocessing
# ==============================================================================
def prepare_set1():
    """
    Prepare Set 1: ML-Dataset.csv (Social Isolation Detection).

    Target: loneliness_score >= 4 -> binary (1 = isolated, 0 = not isolated)
    Features: 20 behavioral indicators from the survey

    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Binary target
        info (dict): Dataset metadata
    """
    print("=" * 70)
    print("SET 1: Loading ML-Dataset.csv")
    print("=" * 70)

    filepath = os.path.join(DATASET_DIR, "ML-Dataset.csv")
    df = pd.read_csv(filepath)
    print(f"  [LOAD] Raw data: {df.shape[0]} rows x {df.shape[1]} columns")

    # Rename columns
    df.rename(columns=COLUMN_MAP, inplace=True)
    print(f"  [RENAME] {len(COLUMN_MAP)} columns renamed to short aliases")

    # --- CGPA Sanitization (regex-based) ---
    def clean_cgpa(value):
        if pd.isna(value):
            return np.nan
        value_str = str(value).strip()
        match = re.findall(r'(\d+\.?\d*)', value_str)
        if match:
            return min(float(match[0]), 4.0)
        return np.nan

    df["cgpa"] = df["cgpa"].apply(clean_cgpa)
    print(f"  [SANITIZE] CGPA cleaned via regex -> range [{df['cgpa'].min():.2f}, {df['cgpa'].max():.2f}]")

    # --- Contextual Imputation by Living Status ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    total_imputed = 0

    for group_name, group_df in df.groupby("living_status"):
        for col in numeric_cols:
            missing_mask = df.loc[group_df.index, col].isna()
            n_missing = missing_mask.sum()
            if n_missing > 0:
                group_median = group_df[col].median()
                df.loc[group_df.index[missing_mask], col] = group_median
                total_imputed += n_missing

    print(f"  [IMPUTE] Contextual imputation: {total_imputed} values filled")

    # --- Outlier Clipping (Z-score |Z| > 3) ---
    for col in ["screen_time", "hours_alone_weekly"]:
        if col in df.columns:
            col_data = df[col].astype(float)
            mean_val, std_val = col_data.mean(), col_data.std()
            if std_val > 0:
                lower, upper = mean_val - 3 * std_val, mean_val + 3 * std_val
                df[col] = col_data.clip(lower=lower, upper=upper)

    print(f"  [CLIP] Outliers clipped via Z-score threshold (|Z| > 3)")

    # --- Select Features ---
    available_features = [f for f in SET1_FEATURES if f in df.columns]
    X = df[available_features].copy()

    # Fill any remaining NaN with column median
    for col in X.columns:
        if X[col].isna().any():
            X[col].fillna(X[col].median(), inplace=True)

    # --- Binary Target: loneliness_score >= 4 ---
    y = (df["loneliness_score"] >= 4).astype(int)

    print(f"  [FEATURES] {len(available_features)} features selected")
    print(f"  [TARGET] loneliness_score >= 4 -> Isolated: {y.sum()}, Not-Isolated: {(y == 0).sum()}")
    print(f"  [READY] X shape: {X.shape}, y shape: {y.shape}")

    info = {
        "name": "Set 1: ML-Dataset.csv",
        "description": "Social Isolation Detection",
        "rows": len(df),
        "features": len(available_features),
        "target": "loneliness_score >= 4 (binary)",
        "positive_count": int(y.sum()),
        "negative_count": int((y == 0).sum()),
        "positive_rate": f"{y.mean() * 100:.1f}%",
    }

    return X, y, info


# ==============================================================================
# SET 2: Stress Dataset.csv + Student Mental health.csv Preprocessing
# ==============================================================================
def prepare_set2():
    """
    Prepare Set 2: Stress Dataset.csv + Student Mental health.csv (Combined).

    Stress Dataset: 843 rows, 25 numeric features, target = stress type
    Student Mental Health: 101 rows, encoded categoricals, target = Depression

    Combined target: distress_binary
      - From Stress Dataset: Distress type = 1, else = 0
      - From Mental Health: Depression = Yes -> 1, else = 0

    Returns:
        X (pd.DataFrame): Combined feature matrix
        y (pd.Series): Binary target
        info (dict): Dataset metadata
    """
    print("\n" + "=" * 70)
    print("SET 2: Loading Stress Dataset.csv + Student Mental health.csv")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Part A: Stress Dataset
    # ------------------------------------------------------------------
    stress_path = os.path.join(DATASET_DIR, "Stress Dataset.csv")
    stress_df = pd.read_csv(stress_path)
    print(f"  [LOAD] Stress Dataset: {stress_df.shape[0]} rows x {stress_df.shape[1]} columns")

    # Rename columns to code-friendly aliases
    stress_target_col = "Which type of stress do you primarily experience?"
    stress_question_cols = [c for c in stress_df.columns if c not in ["Gender", "Age", stress_target_col]]

    stress_rename = {"Gender": "gender", "Age": "age"}
    for i, col in enumerate(stress_question_cols):
        stress_rename[col] = f"stress_q{i+1}"

    stress_df.rename(columns=stress_rename, inplace=True)

    # Create binary target: Distress = 1, else = 0
    stress_df["target"] = stress_df[stress_target_col].apply(
        lambda x: 1 if "Distress" in str(x) and "Eustress" not in str(x) else 0
    )

    # Select features (all numeric columns except target)
    stress_feature_cols = ["gender", "age"] + [f"stress_q{i+1}" for i in range(len(stress_question_cols))]
    stress_features = stress_df[stress_feature_cols].copy()
    stress_target = stress_df["target"].copy()
    stress_features["data_source"] = 0  # marker for later removal

    print(f"  [STRESS] Target distribution: Distress={stress_target.sum()}, Non-Distress={(stress_target == 0).sum()}")

    # ------------------------------------------------------------------
    # Part B: Student Mental Health Dataset
    # ------------------------------------------------------------------
    mental_path = os.path.join(DATASET_DIR, "Student Mental health.csv")
    mental_df = pd.read_csv(mental_path)
    print(f"  [LOAD] Student Mental Health: {mental_df.shape[0]} rows x {mental_df.shape[1]} columns")

    # Encode Gender: Female=1, Male=0
    mental_df["gender"] = mental_df["Choose your gender"].map({"Male": 0, "Female": 1}).fillna(0).astype(int)

    # Encode Age
    mental_df["age"] = mental_df["Age"].fillna(mental_df["Age"].median())

    # Encode CGPA ranges to midpoint values
    cgpa_map = {
        "0 - 1.99": 1.0, "2.00 - 2.49": 2.25, "2.50 - 2.99": 2.75,
        "3.00 - 3.49": 3.25, "3.50 - 4.00": 3.75, "3.50 - 4.00 ": 3.75,
    }
    mental_df["cgpa_encoded"] = mental_df["What is your CGPA?"].map(cgpa_map).fillna(2.75)

    # Encode Year of Study
    year_map = {}
    for y_str in mental_df["Your current year of Study"].dropna().unique():
        digit = re.findall(r'\d+', str(y_str))
        if digit:
            year_map[y_str] = int(digit[0])
    mental_df["year_encoded"] = mental_df["Your current year of Study"].map(year_map).fillna(1).astype(int)

    # Encode Yes/No columns to binary
    yn_map = {"Yes": 1, "No": 0}
    mental_df["has_anxiety"] = mental_df["Do you have Anxiety?"].map(yn_map).fillna(0).astype(int)
    mental_df["has_panic_attack"] = mental_df["Do you have Panic attack?"].map(yn_map).fillna(0).astype(int)
    mental_df["sought_treatment"] = mental_df["Did you seek any specialist for a treatment?"].map(yn_map).fillna(0).astype(int)
    mental_df["is_married"] = mental_df["Marital status"].map(yn_map).fillna(0).astype(int)

    # Target: Depression = Yes -> 1
    mental_df["target"] = mental_df["Do you have Depression?"].map(yn_map).fillna(0).astype(int)

    # Select encoded features
    mental_feature_cols = ["gender", "age", "cgpa_encoded", "year_encoded",
                           "has_anxiety", "has_panic_attack", "sought_treatment", "is_married"]
    mental_features = mental_df[mental_feature_cols].copy()
    mental_target = mental_df["target"].copy()
    mental_features["data_source"] = 1  # marker

    print(f"  [MENTAL] Target distribution: Depression={mental_target.sum()}, No-Depression={(mental_target == 0).sum()}")

    # ------------------------------------------------------------------
    # Part C: Combine Both Datasets
    # ------------------------------------------------------------------
    print("\n  [COMBINE] Merging Stress + Mental Health datasets...")

    # Concatenate row-wise (columns not present in one will be NaN)
    combined_features = pd.concat([stress_features, mental_features], axis=0, ignore_index=True)
    combined_target = pd.concat([stress_target, mental_target], axis=0, ignore_index=True)

    # Remove the source marker
    combined_features.drop(columns=["data_source"], inplace=True)

    # Fill NaN with 0 (absent binary features default to 0 = "No")
    combined_features.fillna(0, inplace=True)

    # Ensure all numeric
    for col in combined_features.columns:
        combined_features[col] = pd.to_numeric(combined_features[col], errors='coerce').fillna(0)

    X = combined_features
    y = combined_target.astype(int)

    print(f"  [COMBINED] Total rows: {len(X)}, Features: {X.shape[1]}")
    print(f"  [TARGET] Distressed/Depressed: {y.sum()}, Healthy: {(y == 0).sum()}")
    print(f"  [READY] X shape: {X.shape}, y shape: {y.shape}")

    info = {
        "name": "Set 2: Stress Dataset + Student Mental Health",
        "description": "Combined Stress & Mental Health Detection",
        "rows": len(X),
        "features": X.shape[1],
        "target": "Distress/Depression (binary)",
        "positive_count": int(y.sum()),
        "negative_count": int((y == 0).sum()),
        "positive_rate": f"{y.mean() * 100:.1f}%",
        "stress_rows": len(stress_df),
        "mental_rows": len(mental_df),
    }

    return X, y, info


# ==============================================================================
# TRAINING & EVALUATION ENGINE
# ==============================================================================
def train_and_evaluate(X, y, set_name, set_label):
    """
    Train all 5 classifiers and compute comprehensive metrics.

    Args:
        X: Feature matrix (pd.DataFrame)
        y: Binary target (pd.Series)
        set_name: Name for display (e.g., "Set 1")
        set_label: Short label for filenames

    Returns:
        results: dict of {model_name: {metrics...}}
    """
    print(f"\n{'=' * 70}")
    print(f"TRAINING & EVALUATION: {set_name}")
    print(f"{'=' * 70}")

    # Train-Test Split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y,
    )
    print(f"  [SPLIT] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n  --- {name} ---")

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Predict probabilities (for ROC curve)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = model.decision_function(X_test_scaled)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        # ROC-AUC
        try:
            auc_score = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc_score = 0.0

        # ROC curve data
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
        except ValueError:
            fpr, tpr = [0, 1], [0, 1]

        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': auc_score,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'fpr': fpr,
            'tpr': tpr,
        }

        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1 Score:  {f1:.4f}")
        print(f"    AUC-ROC:   {auc_score:.4f}")

    return results


# ==============================================================================
# VISUALIZATION: Confusion Matrices
# ==============================================================================
def plot_confusion_matrices(results, set_name, set_label, output_dir):
    """
    Plot confusion matrices for all 5 models in a single figure.

    Args:
        results: dict from train_and_evaluate
        set_name: Display name
        set_label: Filename label (e.g., "set1")
        output_dir: Output directory
    """
    model_names = list(results.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4.5))
    fig.suptitle(f"Confusion Matrices — {set_name}", fontsize=16, fontweight='bold', y=1.02)

    # Professional color palette
    cmap = sns.color_palette("Blues", as_cmap=True)

    for i, name in enumerate(model_names):
        ax = axes[i] if n_models > 1 else axes
        cm = results[name]['confusion_matrix']

        sns.heatmap(
            cm, annot=True, fmt='d', cmap=cmap, ax=ax,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'],
            cbar=False, linewidths=1, linecolor='white',
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(f"{name}\n(Acc: {results[name]['accuracy']:.2%})",
                      fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Predicted", fontsize=10, fontweight='bold')
        ax.set_ylabel("Actual", fontsize=10, fontweight='bold')

    plt.tight_layout()
    set_dir = os.path.join(output_dir, set_label)
    os.makedirs(set_dir, exist_ok=True)
    path = os.path.join(set_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved: {path}")


# ==============================================================================
# VISUALIZATION: ROC Curves
# ==============================================================================
def plot_roc_curves(results, set_name, set_label, output_dir):
    """
    Plot ROC curves for all 5 models overlaid on a single figure.

    Args:
        results: dict from train_and_evaluate
        set_name: Display name
        set_label: Filename label
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Professional color palette
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']

    for i, (name, res) in enumerate(results.items()):
        fpr = res['fpr']
        tpr = res['tpr']
        auc_val = res['roc_auc']

        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2.5,
                label=f"{name} (AUC = {auc_val:.3f})")

    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5,
            label='Random Classifier (AUC = 0.500)')

    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight='bold')
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=13, fontweight='bold')
    ax.set_title(f"ROC Curves — {set_name}", fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    set_dir = os.path.join(output_dir, set_label)
    os.makedirs(set_dir, exist_ok=True)
    path = os.path.join(set_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved: {path}")


# ==============================================================================
# COMPARISON REPORT GENERATION
# ==============================================================================
def save_comparison_report(set1_results, set2_results, set1_info, set2_info, output_dir):
    """
    Generate a comprehensive side-by-side comparison report as a text file.

    Includes:
      - Dataset summaries
      - Per-model metrics for both sets
      - Side-by-side comparison table
      - Classification reports
      - Confusion matrices (text format)
    """
    path = os.path.join(output_dir, "comparison_report.txt")

    with open(path, "w", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ===============================================================
        # HEADER
        # ===============================================================
        f.write("=" * 90 + "\n")
        f.write("  MULTI-DATASET ML PIPELINE — COMPREHENSIVE ACCURACY COMPARISON REPORT\n")
        f.write("=" * 90 + "\n")
        f.write(f"  Generated: {timestamp}\n")
        f.write(f"  Authors:   Muhammad Mobeen, Huzaifa Zaman\n")
        f.write(f"  Course:    Machine Learning (NUTECH, BSAI-2024)\n")
        f.write("=" * 90 + "\n\n")

        # ===============================================================
        # DATASET SUMMARIES
        # ===============================================================
        f.write("+" + "-" * 88 + "+\n")
        f.write("|  DATASET SUMMARIES" + " " * 69 + "|\n")
        f.write("+" + "-" * 88 + "+\n\n")

        for info in [set1_info, set2_info]:
            f.write(f"  {info['name']}\n")
            f.write(f"  {'-' * len(info['name'])}\n")
            f.write(f"    Description:    {info['description']}\n")
            f.write(f"    Total Rows:     {info['rows']}\n")
            f.write(f"    Features:       {info['features']}\n")
            f.write(f"    Target:         {info['target']}\n")
            f.write(f"    Positive Class: {info['positive_count']} ({info['positive_rate']})\n")
            f.write(f"    Negative Class: {info['negative_count']}\n")
            if "stress_rows" in info:
                f.write(f"    Stress Rows:    {info['stress_rows']}\n")
                f.write(f"    Mental H Rows:  {info['mental_rows']}\n")
            f.write("\n")

        # ===============================================================
        # SIDE-BY-SIDE COMPARISON TABLE
        # ===============================================================
        f.write("+" + "-" * 88 + "+\n")
        f.write("|  SIDE-BY-SIDE ACCURACY COMPARISON" + " " * 54 + "|\n")
        f.write("+" + "-" * 88 + "+\n\n")

        # Table header
        header = (
            f"  {'Model':<22} | {'Metric':<10} | "
            f"{'Set 1':>10} | {'Set 2':>10} | {'Diff':>8} | {'Better':>8}"
        )
        separator = "  " + "-" * 82

        f.write(separator + "\n")
        f.write(header + "\n")
        f.write(separator + "\n")

        model_names = list(set1_results.keys())
        metrics_list = [
            ('accuracy', 'Accuracy'),
            ('precision', 'Precision'),
            ('recall', 'Recall'),
            ('f1', 'F1 Score'),
            ('roc_auc', 'AUC-ROC'),
        ]

        for model_name in model_names:
            for j, (metric_key, metric_label) in enumerate(metrics_list):
                val1 = set1_results[model_name][metric_key]
                val2 = set2_results[model_name][metric_key]
                diff = val2 - val1
                better = "Set 2" if val2 > val1 else ("Set 1" if val1 > val2 else "Tie")

                m_name = model_name if j == 0 else ""
                f.write(
                    f"  {m_name:<22} | {metric_label:<10} | "
                    f"{val1:>10.4f} | {val2:>10.4f} | {diff:>+8.4f} | {better:>8}\n"
                )

            f.write(separator + "\n")

        f.write("\n")

        # ===============================================================
        # SUMMARY WINNER TABLE
        # ===============================================================
        f.write("+" + "-" * 88 + "+\n")
        f.write("|  BEST MODEL PER DATASET" + " " * 64 + "|\n")
        f.write("+" + "-" * 88 + "+\n\n")

        for set_label, results, info in [
            ("Set 1", set1_results, set1_info),
            ("Set 2", set2_results, set2_info),
        ]:
            best_acc_model = max(results, key=lambda m: results[m]['accuracy'])
            best_f1_model = max(results, key=lambda m: results[m]['f1'])
            best_auc_model = max(results, key=lambda m: results[m]['roc_auc'])

            f.write(f"  {set_label} ({info['description']})\n")
            f.write(f"    Best Accuracy:  {best_acc_model} -> {results[best_acc_model]['accuracy']:.4f}\n")
            f.write(f"    Best F1 Score:  {best_f1_model} -> {results[best_f1_model]['f1']:.4f}\n")
            f.write(f"    Best AUC-ROC:   {best_auc_model} -> {results[best_auc_model]['roc_auc']:.4f}\n")
            f.write("\n")

        # ===============================================================
        # DETAILED METRICS PER MODEL (SET 1)
        # ===============================================================
        f.write("+" + "-" * 88 + "+\n")
        f.write("|  DETAILED RESULTS — SET 1: ML-Dataset.csv" + " " * 45 + "|\n")
        f.write("+" + "-" * 88 + "+\n\n")

        for model_name, res in set1_results.items():
            f.write(f"  {'─' * 40}\n")
            f.write(f"  {model_name}\n")
            f.write(f"  {'─' * 40}\n")
            f.write(f"    Accuracy:  {res['accuracy']:.4f}\n")
            f.write(f"    Precision: {res['precision']:.4f}\n")
            f.write(f"    Recall:    {res['recall']:.4f}\n")
            f.write(f"    F1 Score:  {res['f1']:.4f}\n")
            f.write(f"    AUC-ROC:   {res['roc_auc']:.4f}\n\n")
            f.write(f"    Confusion Matrix:\n")
            cm = res['confusion_matrix']
            f.write(f"      {'':>15} Pred: 0    Pred: 1\n")
            f.write(f"      {'Actual: 0':>15}  {cm[0][0]:>6}     {cm[0][1]:>6}\n")
            f.write(f"      {'Actual: 1':>15}  {cm[1][0]:>6}     {cm[1][1]:>6}\n\n")
            f.write(f"    Classification Report:\n")
            for line in res['classification_report'].split('\n'):
                f.write(f"      {line}\n")
            f.write("\n")

        # ===============================================================
        # DETAILED METRICS PER MODEL (SET 2)
        # ===============================================================
        f.write("+" + "-" * 88 + "+\n")
        f.write("|  DETAILED RESULTS — SET 2: Stress Dataset + Student Mental Health" + " " * 22 + "|\n")
        f.write("+" + "-" * 88 + "+\n\n")

        for model_name, res in set2_results.items():
            f.write(f"  {'─' * 40}\n")
            f.write(f"  {model_name}\n")
            f.write(f"  {'─' * 40}\n")
            f.write(f"    Accuracy:  {res['accuracy']:.4f}\n")
            f.write(f"    Precision: {res['precision']:.4f}\n")
            f.write(f"    Recall:    {res['recall']:.4f}\n")
            f.write(f"    F1 Score:  {res['f1']:.4f}\n")
            f.write(f"    AUC-ROC:   {res['roc_auc']:.4f}\n\n")
            f.write(f"    Confusion Matrix:\n")
            cm = res['confusion_matrix']
            f.write(f"      {'':>15} Pred: 0    Pred: 1\n")
            f.write(f"      {'Actual: 0':>15}  {cm[0][0]:>6}     {cm[0][1]:>6}\n")
            f.write(f"      {'Actual: 1':>15}  {cm[1][0]:>6}     {cm[1][1]:>6}\n\n")
            f.write(f"    Classification Report:\n")
            for line in res['classification_report'].split('\n'):
                f.write(f"      {line}\n")
            f.write("\n")

        # ===============================================================
        # OVERALL COMPARISON MATRIX
        # ===============================================================
        f.write("+" + "-" * 88 + "+\n")
        f.write("|  OVERALL ACCURACY MATRIX (All Models x Both Sets)" + " " * 37 + "|\n")
        f.write("+" + "-" * 88 + "+\n\n")

        col_width = 14
        row_header_width = 22

        # Table header row
        f.write(f"  {'':>{row_header_width}} |")
        for metric_label in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']:
            f.write(f" {metric_label:^{col_width * 2 + 3}} |")
        f.write("\n")

        f.write(f"  {'':>{row_header_width}} |")
        for _ in range(5):
            f.write(f" {'Set1':>{col_width}} {'Set2':>{col_width}} |")
        f.write("\n")

        f.write("  " + "-" * (row_header_width + 1 + 5 * (col_width * 2 + 4) + 1) + "\n")

        for model_name in model_names:
            r1 = set1_results[model_name]
            r2 = set2_results[model_name]
            f.write(f"  {model_name:>{row_header_width}} |")
            for mk in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                f.write(f" {r1[mk]:>{col_width}.4f} {r2[mk]:>{col_width}.4f} |")
            f.write("\n")

        f.write("\n")

        # ===============================================================
        # FOOTER
        # ===============================================================
        f.write("=" * 90 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 90 + "\n")

    print(f"\n  [REPORT] Comprehensive comparison report saved: {path}")
    return path


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    """
    Main entry point: Run the full multi-dataset ML pipeline.

    1. Prepare Set 1 (ML-Dataset.csv)
    2. Prepare Set 2 (Stress + Mental Health combined)
    3. Train & evaluate 5 classifiers on each set
    4. Generate confusion matrix plots
    5. Generate ROC curve plots
    6. Save side-by-side comparison report
    """
    print("\n" + "#" * 70)
    print("#  MULTI-DATASET ML PIPELINE")
    print("#  Comprehensive Accuracy Comparison Across 2 Dataset Sets")
    print("#" * 70 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ===========================================
    # STEP 1: Prepare Datasets
    # ===========================================
    X1, y1, info1 = prepare_set1()
    X2, y2, info2 = prepare_set2()

    # ===========================================
    # STEP 2: Train & Evaluate on Both Sets
    # ===========================================
    results1 = train_and_evaluate(X1, y1, info1["name"], "set1")
    results2 = train_and_evaluate(X2, y2, info2["name"], "set2")

    # ===========================================
    # STEP 3: Generate Visualizations
    # ===========================================
    print(f"\n{'=' * 70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'=' * 70}")

    plot_confusion_matrices(results1, info1["name"], "set1", OUTPUT_DIR)
    plot_confusion_matrices(results2, info2["name"], "set2", OUTPUT_DIR)
    plot_roc_curves(results1, info1["name"], "set1", OUTPUT_DIR)
    plot_roc_curves(results2, info2["name"], "set2", OUTPUT_DIR)

    # ===========================================
    # STEP 4: Generate Comparison Report
    # ===========================================
    print(f"\n{'=' * 70}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'=' * 70}")

    report_path = save_comparison_report(results1, results2, info1, info2, OUTPUT_DIR)

    # ===========================================
    # FINAL SUMMARY
    # ===========================================
    print(f"\n{'#' * 70}")
    print("#  PIPELINE EXECUTION COMPLETE")
    print(f"{'#' * 70}")

    print(f"\n  {'-' * 50}")
    print(f"  Set 1: {info1['name']}")
    print(f"  {'-' * 50}")
    best1 = max(results1, key=lambda m: results1[m]['accuracy'])
    for name, res in results1.items():
        marker = " * BEST" if name == best1 else ""
        print(f"    {name:<22} Acc: {res['accuracy']:.4f}  F1: {res['f1']:.4f}  AUC: {res['roc_auc']:.4f}{marker}")

    print(f"\n  {'-' * 50}")
    print(f"  Set 2: {info2['name']}")
    print(f"  {'-' * 50}")
    best2 = max(results2, key=lambda m: results2[m]['accuracy'])
    for name, res in results2.items():
        marker = " * BEST" if name == best2 else ""
        print(f"    {name:<22} Acc: {res['accuracy']:.4f}  F1: {res['f1']:.4f}  AUC: {res['roc_auc']:.4f}{marker}")

    print(f"\n  Output files saved in '{OUTPUT_DIR}/':")
    for f_name in sorted(os.listdir(OUTPUT_DIR)):
        f_path = os.path.join(OUTPUT_DIR, f_name)
        size = os.path.getsize(f_path)
        print(f"    {f_name} ({size:,} bytes)")

    print(f"\n  Full comparison report: {report_path}")
    print()


if __name__ == "__main__":
    main()
