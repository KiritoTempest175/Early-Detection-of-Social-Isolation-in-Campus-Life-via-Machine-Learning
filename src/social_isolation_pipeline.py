"""
================================================================================
Early Detection of Social Isolation in Campus Life via Machine Learning
================================================================================
A robust, modular ML pipeline for detecting social isolation anomalies
among university students using behavioral, academic, and social indicators.

Authors: Muhammad Mobeen, Huzaifa Zaman
Course: Machine Learning (NUTECH, BSAI-2024)
Instructor: Lec Faiza Khan

Technical Stack: Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
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

from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Global Constants & Configuration
# ------------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Resolve project root (parent of src/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

# Short aliases for the long survey column names
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

# Features used for modeling (numerical behavioral indicators)
MODELING_FEATURES = [
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
    "Social_Density_Score",
]


# ==============================================================================
# PHASE 1: Data Ingestion & Real-World Preprocessing
# ==============================================================================
class DataPreprocessor:
    """
    Phase 1: Data Ingestion & Real-World Preprocessing Pipeline.

    Handles three critical preprocessing tasks as described in Assignment 3 §3.2-3.3:
      1. Data Sanitization - regex-based CGPA cleaning
      2. Contextual Imputation - median rolling average conditioned on Living Status
      3. Outlier Handling - Z-score thresholding with clip-winsorization

    Reference: Assignment 3, Section 3.2 (Data Ingestion & Real-World Preprocessing Pipeline)
               Assignment 3, Section 3.3 (Handling Common Data Anomalies and Technical Errors)
    """

    def __init__(self, filepath: str):
        """
        Initialize the preprocessor with the path to the CSV dataset.

        Args:
            filepath: Path to ML-Dataset.csv
        """
        self.filepath = filepath
        self.df = None
        self.preprocessing_report = {}

    def load_data(self) -> pd.DataFrame:
        """Load the CSV dataset and apply column renaming."""
        print("=" * 70)
        print("PHASE 1: Data Ingestion & Real-World Preprocessing")
        print("=" * 70)

        self.df = pd.read_csv(self.filepath)
        original_shape = self.df.shape
        print(f"  [LOAD] Dataset loaded: {original_shape[0]} rows x {original_shape[1]} columns")

        # Apply column renaming for code-friendly access
        self.df.rename(columns=COLUMN_MAP, inplace=True)
        print(f"  [RENAME] {len(COLUMN_MAP)} columns renamed to short aliases")

        return self.df

    def sanitize_cgpa(self) -> None:
        """
        Step 1: Data Sanitization - Clean the CGPA column using regex.

        Assignment 3 §3.3.1: 'Students logged CGPA in various mismatched text formats
        (e.g., matching "3.4" with text strings like "3.45/4.0"). Regular expressions were
        engineered to sanitize and cast all performance features into uniform floating-point scales.'

        Strategy:
          - Use regex to extract the first valid numeric value
          - Handle formats like "3.45/4.0" -> 3.45
          - Cap values at the 4.0 scale maximum
          - Fallback to NaN for truly unparseable entries (then imputed in next step)
        """
        print("\n  [SANITIZE] Cleaning CGPA column with regex...")

        def clean_cgpa(value):
            """Extract a clean float from potentially messy CGPA entries."""
            if pd.isna(value):
                return np.nan
            value_str = str(value).strip()
            # Regex: extract the first valid decimal number
            match = re.findall(r'(\d+\.?\d*)', value_str)
            if match:
                numeric_val = float(match[0])
                # Cap at 4.0 scale - any value > 4.0 is likely an error
                return min(numeric_val, 4.0)
            return np.nan

        original_dtype = self.df["cgpa"].dtype
        invalid_before = self.df["cgpa"].isna().sum()

        self.df["cgpa"] = self.df["cgpa"].apply(clean_cgpa)

        invalid_after = self.df["cgpa"].isna().sum()
        print(f"    -> CGPA dtype: {original_dtype} -> float64")
        print(f"    -> Invalid entries before: {invalid_before}, after: {invalid_after}")
        print(f"    -> CGPA range: [{self.df['cgpa'].min():.2f}, {self.df['cgpa'].max():.2f}]")

        self.preprocessing_report["cgpa_invalid_before"] = invalid_before
        self.preprocessing_report["cgpa_invalid_after"] = invalid_after

    def contextual_imputation(self) -> None:
        """
        Step 2: Contextual Imputation - Median rolling average conditioned on Living Status.

        Assignment 3 §3.3.2: 'Instead of dropping columns or rows, which reduces statistical
        power, missing numerical data was imputed using a median rolling average based on the
        student's specific accommodation class (Hostelite vs. Day Scholar).'

        Strategy:
          - Group by Living Status ("Hostelite" vs "Day Scholar")
          - For each numeric column, compute group-conditioned median
          - Fill missing values with the group median
          - This preserves the behavioral differences between hostelites and day scholars
        """
        print("\n  [IMPUTE] Contextual imputation by Living Status group...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        total_imputed = 0

        for group_name, group_df in self.df.groupby("living_status"):
            for col in numeric_cols:
                missing_mask = self.df.loc[group_df.index, col].isna()
                n_missing = missing_mask.sum()
                if n_missing > 0:
                    group_median = group_df[col].median()
                    self.df.loc[group_df.index[missing_mask], col] = group_median
                    total_imputed += n_missing

        print(f"    -> Total values imputed: {total_imputed}")
        print(f"    -> Groups: {self.df['living_status'].value_counts().to_dict()}")
        print(f"    -> Remaining NaN in numeric columns: {self.df[numeric_cols].isna().sum().sum()}")

        self.preprocessing_report["total_imputed"] = total_imputed

    def clip_outliers(self) -> None:
        """
        Step 3: Outlier Handling - Z-score thresholding with clip-winsorization.

        Assignment 3 §3.3.3: 'Screen-time values exceeding 24 hours or study metrics showing
        unrealistic entry durations were caught via Z-score thresholding (|Z| > 3) and capped
        using clip-winsorization to prevent skewing the unsupervised learning architectures.'

        Target columns:
          - Average daily Screen Time on your phone (Hours)? (In weekdays)
          - Total hours spent in personal space/alone on average in a week?

        Strategy:
          - Compute Z-scores for each target column
          - Identify extreme values where |Z| > 3
          - Clip (cap) extreme values to mean ± 3sigma boundaries
          - This preserves row count while preventing model skewing
        """
        print("\n  [CLIP] Outlier handling via Z-score clip-winsorization (|Z| > 3)...")

        target_columns = ["screen_time", "hours_alone_weekly"]

        for col in target_columns:
            if col not in self.df.columns:
                print(f"    -> WARNING: Column '{col}' not found, skipping.")
                continue

            col_data = self.df[col].astype(float)
            mean_val = col_data.mean()
            std_val = col_data.std()

            if std_val == 0:
                print(f"    -> {col}: No variance, skipping.")
                continue

            z_scores = np.abs((col_data - mean_val) / std_val)
            n_outliers = (z_scores > 3).sum()

            # Define clip boundaries: mean ± 3sigma
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val

            # Apply clip-winsorization
            self.df[col] = col_data.clip(lower=lower_bound, upper=upper_bound)

            print(f"    -> {col}:")
            print(f"      Outliers detected: {n_outliers}")
            print(f"      Clip bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"      New range: [{self.df[col].min():.2f}, {self.df[col].max():.2f}]")

            self.preprocessing_report[f"{col}_outliers"] = n_outliers

    def run(self) -> pd.DataFrame:
        """Execute the full Phase 1 preprocessing pipeline in sequence."""
        self.load_data()
        self.sanitize_cgpa()
        self.contextual_imputation()
        self.clip_outliers()

        print(f"\n  [DONE] Phase 1 complete. Clean dataset: {self.df.shape[0]} rows x {self.df.shape[1]} cols")
        print(f"  [REPORT] {self.preprocessing_report}")
        return self.df


# ==============================================================================
# PHASE 2: Feature Engineering
# ==============================================================================
class FeatureEngineer:
    """
    Phase 2: Feature Engineering - Social Density Score construction.

    Engineers a continuous 'Social_Density_Score' feature that captures the ratio
    of communal interactivity versus private isolation.

    Reference: Assignment 1, Section 3 ('calculating Social Density scores')
               Assignment 2, Preliminary Experiments ('Social Density score based on
               the ratio of communal space usage versus private dorm time')
               Assignment 3, Section 3.4 ('calculating Social Density scores')
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with the preprocessed DataFrame.

        Args:
            df: Cleaned DataFrame from Phase 1
        """
        self.df = df.copy()
        self.scaler = MinMaxScaler()

    def build_social_density_score(self) -> None:
        """
        Construct the Social_Density_Score as a ratio/weighted combination of
        Communal Interactivity vs. Private Isolation.

        Communal Interactivity components:
          - meals_with_friends: Number of meals eaten with friends per week
          - common_room_frequency: Frequency of using Common Rooms/Lounges/Library
          - social_outings: Times leaving campus for social reasons

        Private Isolation component:
          - hours_alone_weekly: Total hours spent in personal space/alone per week

        Formula:
          Social_Density_Score = normalized_communal_sum / (1 + normalized_alone_hours)

        Each component is min-max normalized to [0, 1] before combination to ensure
        fair weighting across different scales.
        """
        print("\n" + "=" * 70)
        print("PHASE 2: Feature Engineering")
        print("=" * 70)

        # Communal interactivity indicators
        communal_cols = ["meals_with_friends", "common_room_frequency", "social_outings"]
        # Private isolation indicator
        isolation_col = "hours_alone_weekly"

        print(f"  [ENGINEER] Building Social_Density_Score...")
        print(f"    Communal components: {communal_cols}")
        print(f"    Isolation component: {isolation_col}")

        # Min-max normalize each component to [0, 1]
        communal_data = self.df[communal_cols].values
        isolation_data = self.df[[isolation_col]].values

        communal_normalized = self.scaler.fit_transform(communal_data)
        isolation_normalized = MinMaxScaler().fit_transform(isolation_data)

        # Compute communal sum (average of normalized communal indicators)
        communal_sum = communal_normalized.mean(axis=1)

        # Social Density = communal_sum / (1 + isolation_normalized)
        # Higher score = more socially integrated, lower = more isolated
        isolation_flat = isolation_normalized.flatten()
        self.df["Social_Density_Score"] = communal_sum / (1.0 + isolation_flat)

        print(f"    -> Social_Density_Score statistics:")
        print(f"      Mean:   {self.df['Social_Density_Score'].mean():.4f}")
        print(f"      Std:    {self.df['Social_Density_Score'].std():.4f}")
        print(f"      Min:    {self.df['Social_Density_Score'].min():.4f}")
        print(f"      Max:    {self.df['Social_Density_Score'].max():.4f}")
        print(f"      Median: {self.df['Social_Density_Score'].median():.4f}")

    def run(self) -> pd.DataFrame:
        """Execute the full Phase 2 feature engineering pipeline."""
        self.build_social_density_score()
        print(f"\n  [DONE] Phase 2 complete. Feature set now has {self.df.shape[1]} columns.")
        return self.df


# ==============================================================================
# PHASE 3: Core Unsupervised Modeling
# ==============================================================================
class IsolationModel:
    """
    Phase 3: Core Unsupervised Modeling - K-Means Clustering + Isolation Forest.

    Implements two complementary unsupervised algorithms:
      1. K-Means Clustering (k=3) for behavioral segmentation
      2. Isolation Forest for anomaly detection

    Reference: Assignment 1, Section 3 ('K-Means Clustering', 'Isolation Forests')
               Assignment 2, Preliminary Experiments (3 clusters, 5% contamination)
               Assignment 3, Section 3.4 & 4.1 (Isolation Forest architecture,
               K-Means to segment behavioral patterns)
    """

    def __init__(self, df: pd.DataFrame, features: list):
        """
        Initialize the modeling pipeline.

        Args:
            df: Enriched DataFrame from Phase 2
            features: List of feature column names for modeling
        """
        self.df = df.copy()
        self.features = [f for f in features if f in df.columns]
        self.scaler = StandardScaler()
        self.kmeans = None
        self.isolation_forest = None
        self.X_scaled = None
        self.contamination_rate = None

    def _prepare_features(self) -> np.ndarray:
        """Standardize features for unsupervised learning."""
        print(f"  [FEATURES] Preparing {len(self.features)} features for modeling...")
        X = self.df[self.features].values.astype(float)

        # Handle any remaining NaN by filling with column median
        for i in range(X.shape[1]):
            col_median = np.nanmedian(X[:, i])
            nan_mask = np.isnan(X[:, i])
            X[nan_mask, i] = col_median

        self.X_scaled = self.scaler.fit_transform(X)
        print(f"    -> Feature matrix shape: {self.X_scaled.shape}")
        return self.X_scaled

    def run_kmeans(self, n_clusters: int = 3) -> None:
        """
        Step 1: Behavioral Segmentation via K-Means Clustering.

        Targets 3 distinct clusters as specified in assignments:
          - Cluster A: Highly Social (high Social Density, high communal activity)
          - Cluster B: Academically Focused (high study hours, moderate social)
          - Cluster C: At-Risk (low Social Density, high alone time)

        Cluster labels are assigned post-hoc based on centroid characteristics.

        Args:
            n_clusters: Number of clusters (default=3 per assignment specification)
        """
        print(f"\n  [K-MEANS] Running K-Means Clustering (k={n_clusters})...")

        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_SEED,
            n_init=10,
            max_iter=300,
        )
        cluster_labels = self.kmeans.fit_predict(self.X_scaled)
        self.df["Cluster_ID"] = cluster_labels

        # Compute silhouette score for cluster quality validation
        sil_score = silhouette_score(self.X_scaled, cluster_labels)
        print(f"    -> Silhouette Score: {sil_score:.4f}")

        # Assign semantic labels based on centroid analysis
        self._assign_cluster_labels()

        # Print cluster distribution
        print(f"    -> Cluster distribution:")
        for label, count in self.df["Cluster_Label"].value_counts().items():
            print(f"      {label}: {count} students ({count / len(self.df) * 100:.1f}%)")

    def _assign_cluster_labels(self) -> None:
        """
        Assign semantic labels to clusters based on centroid characteristics.

        Uses Social_Density_Score and daily_self_study_hours from centroids to
        determine which cluster corresponds to which behavioral profile.
        """
        # Get feature indices for key indicators
        sds_idx = self.features.index("Social_Density_Score")
        study_idx = self.features.index("daily_self_study_hours")

        centroids = self.kmeans.cluster_centers_
        cluster_profiles = {}

        for i in range(len(centroids)):
            cluster_profiles[i] = {
                "social_density": centroids[i][sds_idx],
                "study_hours": centroids[i][study_idx],
            }

        # Sort clusters by social density to assign labels
        sorted_clusters = sorted(cluster_profiles.items(), key=lambda x: x[1]["social_density"])

        label_map = {}
        # Lowest Social Density -> At-Risk
        label_map[sorted_clusters[0][0]] = "At-Risk"
        # Highest Social Density -> Highly Social
        label_map[sorted_clusters[-1][0]] = "Highly Social"
        # Middle -> Academically Focused
        label_map[sorted_clusters[1][0]] = "Academically Focused"

        self.df["Cluster_Label"] = self.df["Cluster_ID"].map(label_map)

        print(f"    -> Cluster label mapping: {label_map}")
        for cid, profile in cluster_profiles.items():
            print(
                f"      Cluster {cid} ({label_map[cid]}): "
                f"SDS={profile['social_density']:.3f}, "
                f"Study={profile['study_hours']:.3f}"
            )

    def compute_contamination_rate(self) -> float:
        """
        Compute the Isolation Forest contamination parameter based on the
        distribution of self-reported loneliness scores.

        Assignment 3 §4.1: The contamination rate is calibrated based on the
        proportion of students reporting high loneliness (score >= 4 on 1-5 scale).

        Returns:
            Contamination rate as a float between 0.01 and 0.5
        """
        loneliness_col = "loneliness_score"
        if loneliness_col in self.df.columns:
            # Students scoring >= 4 on the 1-5 loneliness scale are considered
            # to be reporting significant isolation
            high_loneliness = (self.df[loneliness_col] >= 4).sum()
            total = len(self.df)
            rate = high_loneliness / total

            # Clamp to a reasonable range for Isolation Forest
            rate = np.clip(rate, 0.01, 0.3)
            print(f"    -> High loneliness (>=4): {high_loneliness}/{total} = {rate:.4f}")
        else:
            rate = 0.05  # Default fallback as mentioned in Assignment 2
            print(f"    -> Using default contamination rate: {rate}")

        self.contamination_rate = rate
        return rate

    def run_isolation_forest(self) -> None:
        """
        Step 2: Anomaly Detection via Isolation Forest.

        Assignment 3 §4.1: 'The core predictive mechanism utilizes an Isolation Forest
        architecture via Scikit-Learn. Isolation Forests optimize this process by isolating
        anomalies directly using random partitioning trees. Because isolated individuals
        exhibit behavioral paths that are statistical outliers, their samples require
        significantly fewer splits to isolate in a tree structure.'

        The contamination parameter is dynamically calibrated based on self-reported
        loneliness distribution rather than using an arbitrary fixed value.
        """
        print(f"\n  [ISOLATION FOREST] Running anomaly detection...")

        contamination = self.compute_contamination_rate()

        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=RANDOM_SEED,
            max_samples="auto",
        )

        # Fit and predict: -1 = anomaly (isolated), 1 = normal
        predictions = self.isolation_forest.fit_predict(self.X_scaled)
        self.df["Anomaly_Flag"] = predictions
        self.df["Anomaly_Flag_Binary"] = (predictions == -1).astype(int)

        # Get anomaly scores (lower = more anomalous)
        self.df["Anomaly_Score"] = self.isolation_forest.decision_function(self.X_scaled)

        n_anomalies = (predictions == -1).sum()
        print(f"    -> Contamination rate: {contamination:.4f}")
        print(f"    -> Anomalies detected: {n_anomalies}/{len(self.df)} ({n_anomalies / len(self.df) * 100:.1f}%)")
        print(f"    -> Anomaly score range: [{self.df['Anomaly_Score'].min():.4f}, {self.df['Anomaly_Score'].max():.4f}]")

    def run(self) -> pd.DataFrame:
        """Execute the full Phase 3 modeling pipeline."""
        print("\n" + "=" * 70)
        print("PHASE 3: Core Unsupervised Modeling")
        print("=" * 70)

        self._prepare_features()
        self.run_kmeans(n_clusters=3)
        self.run_isolation_forest()

        print(f"\n  [DONE] Phase 3 complete. Added Cluster_Label, Anomaly_Flag, Anomaly_Score columns.")
        return self.df


# ==============================================================================
# PHASE 4: False Positive Mitigation
# ==============================================================================
class FalsePositiveFilter:
    """
    Phase 4: False Positive Mitigation - The High-Academic Exception Filter.

    Implements a programmatic, conditional filtering matrix post-prediction to
    prevent healthy introverts from being flagged as At-Risk.

    Reference: Assignment 3, Section 4.2 ('Mitigating False Positives: Introversion
               vs. Negative Isolation')
               - 'The High-Academic Exception Filter: The system maps the isolation
                 anomaly score against the intersection of Daily Self-Study Hours,
                 Library Durations, and CGPA.'
               - 'If a student exhibits low social dining and zero society interaction
                 but shows high academic engagement alongside a stable, strong GPA
                 profile, the model shifts their classification from At-Risk to
                 Academically Focused Overachiever.'
               - 'Genuine social isolation is prioritized when a low communal presence
                 matches drop-offs in class attendance, elevated screen time, high
                 self-reported stress levels, and falling subjective mood scales.'
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with the modeled DataFrame.

        Args:
            df: DataFrame with cluster labels and anomaly flags from Phase 3
        """
        self.df = df.copy()
        self.reclassified_count = 0
        self.confirmed_atrisk_count = 0

    def apply_high_academic_exception(self) -> None:
        """
        Cross-Reference Logic (Assignment 3 §4.2):

        IF the model flags a student as an anomaly (Isolation Forest = -1)
           AND they exhibit low communal presence (low Social Density)
        AND they have:
           - High Average Daily Self-Study Hours (>= 75th percentile)
           - Stable Current CGPA (>= 3.0)
           - Good class attendance (>= 4 on 1-5 scale)
        THEN reclassify from "At-Risk" -> "Academically Focused Overachiever"
        """
        print("\n" + "=" * 70)
        print("PHASE 4: False Positive Mitigation (High-Academic Exception Filter)")
        print("=" * 70)

        # Compute thresholds
        study_75th = self.df["daily_self_study_hours"].quantile(0.75)
        cgpa_threshold = 3.0
        attendance_threshold = 4
        social_density_median = self.df["Social_Density_Score"].median()

        print(f"  [THRESHOLDS]")
        print(f"    Self-study hours >= 75th percentile: {study_75th:.2f}")
        print(f"    CGPA threshold: >= {cgpa_threshold}")
        print(f"    Attendance threshold: >= {attendance_threshold}")
        print(f"    Social Density median (for 'low communal'): {social_density_median:.4f}")

        # Initialize final risk category with cluster label
        self.df["Final_Risk_Category"] = self.df["Cluster_Label"].copy()

        # Identify flagged anomalies
        anomaly_mask = self.df["Anomaly_Flag"] == -1

        # High-academic exception: anomaly + low social + high academics
        exception_mask = (
            anomaly_mask
            & (self.df["Social_Density_Score"] < social_density_median)
            & (self.df["daily_self_study_hours"] >= study_75th)
            & (self.df["cgpa"] >= cgpa_threshold)
            & (self.df["class_attendance"] >= attendance_threshold)
        )

        self.reclassified_count = exception_mask.sum()
        self.df.loc[exception_mask, "Final_Risk_Category"] = "Academically Focused Overachiever"

        print(f"\n  [EXCEPTION FILTER] Applied High-Academic Exception:")
        print(f"    -> Students reclassified to 'Overachiever': {self.reclassified_count}")

    def confirm_genuine_isolation(self) -> None:
        """
        Genuine Isolation Confirmation (Assignment 3 §4.2):

        Maintain the "At-Risk" flag ONLY IF low communal presence correlates with:
          - Elevated screen time (>= 75th percentile)
          - High Perceived Academic Workload Stress (>= 7 on 1-10 scale)
          - Low overall mood rating (<= 3 on 1-10 scale)

        Students flagged as anomalies who don't meet either the exception filter
        OR the genuine isolation criteria get a softer "Monitor" label.
        """
        # Compute thresholds for genuine isolation
        screen_time_75th = self.df["screen_time"].quantile(0.75)
        stress_threshold = 7
        mood_threshold = 3

        print(f"\n  [GENUINE ISOLATION] Confirming At-Risk flags:")
        print(f"    Screen time >= 75th pctl: {screen_time_75th:.2f}")
        print(f"    Stress threshold: >= {stress_threshold}")
        print(f"    Mood threshold: <= {mood_threshold}")

        anomaly_mask = self.df["Anomaly_Flag"] == -1
        not_overachiever = self.df["Final_Risk_Category"] != "Academically Focused Overachiever"
        social_density_median = self.df["Social_Density_Score"].median()

        # Genuine isolation: anomaly + low social + high stress indicators
        genuine_isolation_mask = (
            anomaly_mask
            & not_overachiever
            & (self.df["Social_Density_Score"] < social_density_median)
            & (self.df["screen_time"] >= screen_time_75th)
            & (self.df["workload_stress"] >= stress_threshold)
            & (self.df["mood_rating"] <= mood_threshold)
        )

        self.confirmed_atrisk_count = genuine_isolation_mask.sum()
        self.df.loc[genuine_isolation_mask, "Final_Risk_Category"] = "Confirmed At-Risk"

        # Remaining anomalies that don't meet strict criteria -> "Monitor"
        remaining_anomalies = (
            anomaly_mask
            & not_overachiever
            & ~genuine_isolation_mask
        )
        monitor_count = remaining_anomalies.sum()
        self.df.loc[remaining_anomalies, "Final_Risk_Category"] = "Monitor"

        print(f"    -> Confirmed At-Risk (genuine isolation): {self.confirmed_atrisk_count}")
        print(f"    -> Reclassified to Monitor (soft flag): {monitor_count}")

    def run(self) -> pd.DataFrame:
        """Execute the full Phase 4 false positive mitigation pipeline."""
        self.apply_high_academic_exception()
        self.confirm_genuine_isolation()

        # Final distribution
        print(f"\n  [FINAL DISTRIBUTION]")
        for cat, count in self.df["Final_Risk_Category"].value_counts().items():
            print(f"    {cat}: {count} ({count / len(self.df) * 100:.1f}%)")

        print(f"\n  [DONE] Phase 4 complete. Final_Risk_Category column added.")
        return self.df


# ==============================================================================
# PHASE 5: Evaluation & Validation Suite
# ==============================================================================
class EvaluationSuite:
    """
    Phase 5: Evaluation & Validation Suite.

    Since isolation is a minority class, raw accuracy is highly misleading.
    This suite evaluates the model using:
      1. Precision-Recall (PR) Curves
      2. Area Under the ROC Curve (AUC-ROC)
      3. Mean Absolute Error (MAE)
      4. Silhouette Coefficients
      5. Stratified K-Fold Cross-Validation

    Reference: Assignment 1, Section 4 (PR Curves, MAE, Silhouette Coefficients)
               Assignment 2, Problems §2 ('shifting evaluation metrics to PR Curves')
               Assignment 3, Section 2 ('AUC-ROC and PR Curves alongside Silhouette
               Coefficients to rigorously measure the model's true capability')
    """

    def __init__(self, df: pd.DataFrame, X_scaled: np.ndarray, features: list, output_dir: str = "outputs"):
        """
        Initialize the evaluation suite.

        Args:
            df: Fully processed DataFrame with all model outputs
            X_scaled: Scaled feature matrix used for modeling
            features: List of feature names
            output_dir: Directory for saving plots and reports
        """
        self.df = df
        self.X_scaled = X_scaled
        self.features = features
        self.output_dir = output_dir
        self.metrics = {}

        os.makedirs(self.output_dir, exist_ok=True)

    def _get_ground_truth_binary(self) -> np.ndarray:
        """
        Create a binary ground truth label from self-reported loneliness.
        Loneliness score >= 4 (on 1-5 scale) = isolated (positive class).
        """
        return (self.df["loneliness_score"] >= 4).astype(int).values

    def evaluate_precision_recall(self) -> None:
        """
        Metric 1: Precision-Recall Curves.

        Since isolation is a minority class, PR curves are more informative than
        simple accuracy. This measures the tradeoff between correctly identifying
        isolated students (recall) and avoiding false alarms (precision).
        """
        print("\n  [PR CURVE] Computing Precision-Recall curve...")

        y_true = self._get_ground_truth_binary()
        # Use negative anomaly scores (lower = more anomalous -> higher probability of isolation)
        y_scores = -self.df["Anomaly_Score"].values

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        self.metrics["average_precision"] = avg_precision

        print(f"    -> Average Precision (AP): {avg_precision:.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(recall, precision, color="#E74C3C", linewidth=2.5, label=f"PR Curve (AP = {avg_precision:.3f})")
        ax.fill_between(recall, precision, alpha=0.15, color="#E74C3C")
        ax.set_xlabel("Recall (Sensitivity)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Precision", fontsize=13, fontweight="bold")
        ax.set_title("Precision-Recall Curve - Social Isolation Detection", fontsize=15, fontweight="bold")
        ax.legend(loc="upper right", fontsize=12)
        ax.set_xlim([0.0, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)

        # Add baseline (random classifier)
        baseline = y_true.sum() / len(y_true)
        ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1, label=f"Baseline ({baseline:.3f})")
        ax.legend(loc="upper right", fontsize=12)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "pr_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    -> Saved: {path}")

    def evaluate_roc_auc(self) -> None:
        """
        Metric 2: Area Under the ROC Curve (AUC-ROC).

        Assignment 3 §2: 'We integrated Area Under the ROC Curve (AUC-ROC)...
        to rigorously measure the model's true capability in isolating the
        minority class (isolated students) without inflating accuracy scores.'
        """
        print("\n  [ROC-AUC] Computing ROC curve...")

        y_true = self._get_ground_truth_binary()
        y_scores = -self.df["Anomaly_Score"].values

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        self.metrics["roc_auc"] = roc_auc

        print(f"    -> AUC-ROC: {roc_auc:.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(fpr, tpr, color="#3498DB", linewidth=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.15, color="#3498DB")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random Classifier")
        ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=13, fontweight="bold")
        ax.set_title("ROC Curve - Social Isolation Detection", fontsize=15, fontweight="bold")
        ax.legend(loc="lower right", fontsize=12)
        ax.set_xlim([0.0, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "roc_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    -> Saved: {path}")

    def evaluate_mae(self) -> None:
        """
        Metric 3: Mean Absolute Error (MAE).

        Compares the model's Isolation Score (inverted anomaly score normalized to
        1-5 scale) against the student's self-reported social wellness rating.

        Assignment 1: 'Mean Absolute Error (MAE): To measure the distance between
        reported mood from our survey.'
        """
        print("\n  [MAE] Computing Mean Absolute Error...")

        # Normalize anomaly scores to a 1-5 scale for comparison with wellness rating
        anomaly_scores = self.df["Anomaly_Score"].values
        # Invert: more anomalous (lower score) -> higher isolation (higher value)
        isolation_scores_raw = -anomaly_scores
        # Min-max normalize to [1, 5]
        min_s = isolation_scores_raw.min()
        max_s = isolation_scores_raw.max()
        if max_s - min_s > 0:
            isolation_normalized = 1 + 4 * (isolation_scores_raw - min_s) / (max_s - min_s)
        else:
            isolation_normalized = np.full_like(isolation_scores_raw, 3.0)

        # Compare against self-reported social wellness
        y_true = self.df["social_wellness_rating"].values
        mae = mean_absolute_error(y_true, isolation_normalized)
        self.metrics["mae"] = mae

        print(f"    -> MAE (Model Isolation Score vs. Self-Reported Wellness): {mae:.4f}")
        print(f"    -> Interpretation: Average {mae:.2f} point difference on 1-5 scale")

    def evaluate_silhouette(self) -> None:
        """
        Metric 4: Silhouette Coefficients.

        Validates the cohesion and separation of the K-Means behavioral clusters.

        Assignment 1: 'Silhouette Coefficients: To validate the consistency and
        separation of our behavioral clusters.'
        """
        print("\n  [SILHOUETTE] Computing Silhouette analysis...")

        cluster_labels = self.df["Cluster_ID"].values
        sil_avg = silhouette_score(self.X_scaled, cluster_labels)
        sil_samples = silhouette_samples(self.X_scaled, cluster_labels)
        self.metrics["silhouette_avg"] = sil_avg

        print(f"    -> Average Silhouette Score: {sil_avg:.4f}")
        print(f"    -> Interpretation: ", end="")
        if sil_avg > 0.5:
            print("Strong cluster structure")
        elif sil_avg > 0.25:
            print("Reasonable cluster structure")
        else:
            print("Weak cluster structure (overlapping clusters)")

        # Per-cluster silhouette scores
        for i in sorted(self.df["Cluster_ID"].unique()):
            cluster_sil = sil_samples[cluster_labels == i].mean()
            label = self.df.loc[self.df["Cluster_ID"] == i, "Cluster_Label"].iloc[0]
            print(f"      Cluster {i} ({label}): {cluster_sil:.4f}")

        # Plot silhouette analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Left: Silhouette plot
        y_lower = 10
        n_clusters = len(self.df["Cluster_ID"].unique())
        colors = ["#E74C3C", "#3498DB", "#2ECC71"]

        for i in sorted(self.df["Cluster_ID"].unique()):
            ith_cluster_sil = sil_samples[cluster_labels == i]
            ith_cluster_sil.sort()
            size_cluster_i = ith_cluster_sil.shape[0]
            y_upper = y_lower + size_cluster_i

            color = colors[i % len(colors)]
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_sil,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            label = self.df.loc[self.df["Cluster_ID"] == i, "Cluster_Label"].iloc[0]
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, label, fontsize=10)
            y_lower = y_upper + 10

        ax1.set_xlabel("Silhouette Coefficient", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Students (sorted by cluster)", fontsize=12, fontweight="bold")
        ax1.set_title("Silhouette Analysis", fontsize=14, fontweight="bold")
        ax1.axvline(x=sil_avg, color="red", linestyle="--", linewidth=1.5,
                     label=f"Avg Score: {sil_avg:.3f}")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Right: Cluster distribution bar chart
        cluster_counts = self.df["Cluster_Label"].value_counts()
        bars = ax2.bar(cluster_counts.index, cluster_counts.values, color=colors[:n_clusters], alpha=0.8, edgecolor="black")
        ax2.set_xlabel("Cluster", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Number of Students", fontsize=12, fontweight="bold")
        ax2.set_title("Cluster Distribution", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        # Add count labels on bars
        for bar, count in zip(bars, cluster_counts.values):
            ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 2,
                     str(count), ha="center", va="bottom", fontweight="bold", fontsize=12)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "silhouette_analysis.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    -> Saved: {path}")

    def run_stratified_kfold(self) -> None:
        """
        Safeguard: Stratified K-Fold Cross-Validation (k=5).

        Ensures the Isolation Forest generalizes well and neither overfits to survey
        noise nor underfits behavioral signals. Uses self-reported loneliness as
        the stratification label.
        """
        print("\n  [CROSS-VALIDATION] Stratified 5-Fold Cross-Validation...")

        y_true = self._get_ground_truth_binary()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        fold_aucs = []
        fold_aps = []

        contamination = (y_true == 1).sum() / len(y_true)
        contamination = np.clip(contamination, 0.01, 0.3)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(self.X_scaled, y_true)):
            X_train, X_test = self.X_scaled[train_idx], self.X_scaled[test_idx]
            y_test = y_true[test_idx]

            iso_forest = IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=RANDOM_SEED,
            )
            iso_forest.fit(X_train)

            scores = -iso_forest.decision_function(X_test)

            try:
                auc = roc_auc_score(y_test, scores)
                ap = average_precision_score(y_test, scores)
                fold_aucs.append(auc)
                fold_aps.append(ap)
                print(f"    Fold {fold_idx + 1}: AUC-ROC={auc:.4f}, AP={ap:.4f}")
            except ValueError:
                print(f"    Fold {fold_idx + 1}: Skipped (single class in fold)")

        if fold_aucs:
            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            mean_ap = np.mean(fold_aps)
            std_ap = np.std(fold_aps)

            self.metrics["cv_auc_mean"] = mean_auc
            self.metrics["cv_auc_std"] = std_auc
            self.metrics["cv_ap_mean"] = mean_ap
            self.metrics["cv_ap_std"] = std_ap

            print(f"    -> Mean AUC-ROC: {mean_auc:.4f} ± {std_auc:.4f}")
            print(f"    -> Mean AP:      {mean_ap:.4f} ± {std_ap:.4f}")
            print(f"    -> Interpretation: ", end="")
            if std_auc < 0.05:
                print("Low variance - model generalizes well.")
            else:
                print("Higher variance - some sensitivity to data splits.")

    def save_summary(self) -> None:
        """Save a comprehensive text summary of all evaluation metrics."""
        path = os.path.join(self.output_dir, "pipeline_summary.txt")

        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("SOCIAL ISOLATION DETECTION - EVALUATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write("Dataset: ML-Dataset.csv\n")
            f.write(f"Total Students: {len(self.df)}\n")
            f.write(f"Features Used: {len(self.features)}\n\n")

            f.write("-" * 40 + "\n")
            f.write("MODEL EVALUATION METRICS\n")
            f.write("-" * 40 + "\n\n")

            for key, value in self.metrics.items():
                f.write(f"  {key}: {value:.4f}\n")

            f.write("\n")
            f.write("-" * 40 + "\n")
            f.write("FINAL RISK DISTRIBUTION\n")
            f.write("-" * 40 + "\n\n")

            for cat, count in self.df["Final_Risk_Category"].value_counts().items():
                f.write(f"  {cat}: {count} ({count / len(self.df) * 100:.1f}%)\n")

            f.write("\n")
            f.write("-" * 40 + "\n")
            f.write("CLUSTER DISTRIBUTION\n")
            f.write("-" * 40 + "\n\n")

            for label, count in self.df["Cluster_Label"].value_counts().items():
                f.write(f"  {label}: {count} ({count / len(self.df) * 100:.1f}%)\n")

        print(f"\n  [SUMMARY] Saved: {path}")

    def run(self) -> dict:
        """Execute the full Phase 5 evaluation suite."""
        print("\n" + "=" * 70)
        print("PHASE 5: Evaluation & Validation Suite")
        print("=" * 70)
        print("  Note: Raw accuracy is NOT used - isolation is a minority class.")

        self.evaluate_precision_recall()
        self.evaluate_roc_auc()
        self.evaluate_mae()
        self.evaluate_silhouette()
        self.run_stratified_kfold()
        self.save_summary()

        print(f"\n  [DONE] Phase 5 complete. All metrics computed and plots saved.")
        return self.metrics


# ==============================================================================
# PIPELINE ORCHESTRATOR
# ==============================================================================
class PipelineOrchestrator:
    """
    Master orchestrator that chains all 5 phases of the ML pipeline.

    Pipeline Flow:
      Phase 1: DataPreprocessor  -> Cleaned DataFrame
      Phase 2: FeatureEngineer   -> Enriched DataFrame (+ Social Density Score)
      Phase 3: IsolationModel    -> Clustered + Anomaly-flagged DataFrame
      Phase 4: FalsePositiveFilter -> Refined risk labels
      Phase 5: EvaluationSuite   -> Metrics, plots, and summary report
    """

    def __init__(self, csv_path: str, output_dir: str = "outputs"):
        """
        Initialize the pipeline orchestrator.

        Args:
            csv_path: Path to the ML-Dataset.csv file
            output_dir: Directory for saving all output files
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.df = None
        self.model = None
        self.metrics = None

    def run(self) -> None:
        """Execute the full end-to-end ML pipeline."""
        print("\n" + "#" * 70)
        print("#  SOCIAL ISOLATION DETECTION - ML PIPELINE")
        print("#  Early Detection of Social Isolation in Campus Life")
        print("#" * 70)

        # -- Phase 1: Data Ingestion & Preprocessing --
        preprocessor = DataPreprocessor(self.csv_path)
        self.df = preprocessor.run()

        # -- Phase 2: Feature Engineering --
        engineer = FeatureEngineer(self.df)
        self.df = engineer.run()

        # -- Phase 3: Core Unsupervised Modeling --
        self.model = IsolationModel(self.df, MODELING_FEATURES)
        self.df = self.model.run()

        # -- Phase 4: False Positive Mitigation --
        fp_filter = FalsePositiveFilter(self.df)
        self.df = fp_filter.run()

        # -- Phase 5: Evaluation & Validation --
        evaluator = EvaluationSuite(
            df=self.df,
            X_scaled=self.model.X_scaled,
            features=self.model.features,
            output_dir=self.output_dir,
        )
        self.metrics = evaluator.run()

        # -- Save Final Results --
        self._save_results()

        # -- Final Summary --
        self._print_final_summary()

    def _save_results(self) -> None:
        """Save the final annotated dataset with all predictions."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Select key columns for the output CSV
        output_cols = [
            "name", "university", "semester", "living_status",
            "cgpa", "class_attendance", "daily_self_study_hours",
            "meals_with_friends", "hours_alone_weekly", "common_room_frequency",
            "screen_time", "workload_stress", "mood_rating",
            "loneliness_score", "social_wellness_rating", "peer_satisfaction",
            "Social_Density_Score", "Cluster_ID", "Cluster_Label",
            "Anomaly_Flag", "Anomaly_Score", "Final_Risk_Category",
        ]

        # Only include columns that exist
        existing_cols = [c for c in output_cols if c in self.df.columns]
        result_df = self.df[existing_cols].copy()

        path = os.path.join(self.output_dir, "final_results.csv")
        result_df.to_csv(path, index=False)
        print(f"\n  [SAVE] Final results saved: {path}")
        print(f"    -> {result_df.shape[0]} students x {result_df.shape[1]} columns")

    def _print_final_summary(self) -> None:
        """Print a concise final summary of the pipeline execution."""
        print("\n" + "#" * 70)
        print("#  PIPELINE EXECUTION COMPLETE")
        print("#" * 70)
        print(f"\n  Dataset:  {self.csv_path}")
        print(f"  Students: {len(self.df)}")
        print(f"  Features: {len(self.model.features)}")
        print(f"\n  -- Key Metrics --")

        if self.metrics:
            for key, value in self.metrics.items():
                print(f"    {key}: {value:.4f}")

        print(f"\n  -- Risk Distribution --")
        for cat, count in self.df["Final_Risk_Category"].value_counts().items():
            pct = count / len(self.df) * 100
            print(f"    {cat}: {count} ({pct:.1f}%)")

        print(f"\n  -- Output Files --")
        for f in os.listdir(self.output_dir):
            fpath = os.path.join(self.output_dir, f)
            size = os.path.getsize(fpath)
            print(f"    {f} ({size:,} bytes)")



# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    """
    Main entry point for the Social Isolation Detection ML Pipeline.

    Usage:
        python src/social_isolation_pipeline.py
    """
    csv_path = os.path.join(_PROJECT_ROOT, "dataset", "ML-Dataset.csv")
    output_dir = os.path.join(_PROJECT_ROOT, "outputs")

    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at '{csv_path}'")
        print("Please ensure dataset/ML-Dataset.csv exists.")
        return

    pipeline = PipelineOrchestrator(csv_path=csv_path, output_dir=output_dir)
    pipeline.run()


if __name__ == "__main__":
    main()
