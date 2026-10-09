"""
ML Inference Engine for the Campus AI Backend.

Adapts the full social_isolation_pipeline.py (Phases 1-4) for real-time,
single-student inference.  All preprocessing, feature engineering,
clustering, anomaly detection, and false-positive filtering logic is
replicated exactly from the original pipeline.
"""

import os
import re
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from models import AssessmentInput, PredictionResponse, RecommendationPayload

warnings.filterwarnings("ignore")
logger = logging.getLogger("ml_engine")

# ---------------------------------------------------------------------------
# Constants — identical to social_isolation_pipeline.py
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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

MODELING_FEATURES: List[str] = [
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

# Columns used to build Social Density Score
COMMUNAL_COLS = ["meals_with_friends", "common_room_frequency", "social_outings"]
ISOLATION_COL = "hours_alone_weekly"


class MLEngine:
    """
    Self-contained ML inference engine.

    On initialisation it:
      1. Loads and preprocesses the campus-wide dataset (Phase 1).
      2. Engineers the Social Density Score feature (Phase 2).
      3. Fits StandardScaler, KMeans(k=3), and IsolationForest(n=200)
         on the full dataset (Phase 3).
      4. Runs the false-positive filter to discover cluster-label
         mappings and thresholds (Phase 4).
      5. Computes campus-wide baselines for the radar chart.

    After init, ``predict(input)`` scores a single student in < 5 ms.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, dataset_path: str) -> None:
        logger.info("Initialising MLEngine from %s", dataset_path)

        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at '{dataset_path}'. "
                "Ensure ML-Dataset.csv is accessible."
            )

        # ---- Phase 1: Data Ingestion & Preprocessing --------------------
        self.df = self._phase1_preprocess(dataset_path)

        # ---- Phase 2: Feature Engineering (SDS) -------------------------
        self.df, self.communal_scaler, self.isolation_scaler = (
            self._phase2_feature_engineering(self.df)
        )

        # ---- Phase 3: Core Unsupervised Modeling ------------------------
        (
            self.standard_scaler,
            self.kmeans,
            self.isolation_forest,
            self.contamination_rate,
        ) = self._phase3_modeling(self.df)

        # ---- Phase 4: Derive cluster-label map & thresholds -------------
        self.cluster_label_map = self._derive_cluster_label_map()
        self.df["Cluster_Label"] = self.df["Cluster_ID"].map(self.cluster_label_map)

        # False-positive thresholds (computed from training set)
        self.study_75th = self.df["daily_self_study_hours"].quantile(0.75)
        self.cgpa_threshold = 3.0
        self.attendance_threshold = 4
        self.social_density_median = self.df["Social_Density_Score"].median()
        self.screen_time_75th = self.df["screen_time"].quantile(0.75)
        self.stress_threshold = 7
        self.mood_threshold = 3

        # ---- Campus baselines for radar chart ---------------------------
        self.campus_baseline = self._compute_campus_baselines()

        logger.info("MLEngine ready. %d students in training set.", len(self.df))

    # ------------------------------------------------------------------
    # Phase 1  –  Data Ingestion & Preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _phase1_preprocess(filepath: str) -> pd.DataFrame:
        """Replicate Phase 1 of social_isolation_pipeline.py."""
        df = pd.read_csv(filepath)
        df.rename(columns=COLUMN_MAP, inplace=True)

        # Step 1: CGPA sanitisation (regex)
        def clean_cgpa(value: Any) -> float:
            if pd.isna(value):
                return np.nan
            value_str = str(value).strip()
            match = re.findall(r"(\d+\.?\d*)", value_str)
            if match:
                return min(float(match[0]), 4.0)
            return np.nan

        df["cgpa"] = df["cgpa"].apply(clean_cgpa)

        # Step 2: Contextual imputation by living_status group median
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for _group_name, group_df in df.groupby("living_status"):
            for col in numeric_cols:
                missing_mask = df.loc[group_df.index, col].isna()
                n_missing = missing_mask.sum()
                if n_missing > 0:
                    group_median = group_df[col].median()
                    df.loc[group_df.index[missing_mask], col] = group_median

        # Step 3: Z-score clip-winsorisation for screen_time & hours_alone
        for col in ["screen_time", "hours_alone_weekly"]:
            if col not in df.columns:
                continue
            col_data = df[col].astype(float)
            mean_val = col_data.mean()
            std_val = col_data.std()
            if std_val == 0:
                continue
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            df[col] = col_data.clip(lower=lower_bound, upper=upper_bound)

        return df

    # ------------------------------------------------------------------
    # Phase 2  –  Feature Engineering (Social Density Score)
    # ------------------------------------------------------------------
    @staticmethod
    def _phase2_feature_engineering(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, MinMaxScaler, MinMaxScaler]:
        """
        Build Social_Density_Score exactly as in the pipeline:
          SDS = mean(normalised communal cols) / (1 + normalised isolation col)
        Returns the dataframe plus fitted scalers for later single-row use.
        """
        df = df.copy()

        communal_scaler = MinMaxScaler()
        isolation_scaler = MinMaxScaler()

        communal_data = df[COMMUNAL_COLS].values
        isolation_data = df[[ISOLATION_COL]].values

        communal_normalised = communal_scaler.fit_transform(communal_data)
        isolation_normalised = isolation_scaler.fit_transform(isolation_data)

        communal_mean = communal_normalised.mean(axis=1)
        isolation_flat = isolation_normalised.flatten()

        df["Social_Density_Score"] = communal_mean / (1.0 + isolation_flat)

        return df, communal_scaler, isolation_scaler

    # ------------------------------------------------------------------
    # Phase 3  –  Core Unsupervised Modeling
    # ------------------------------------------------------------------
    def _phase3_modeling(
        self, df: pd.DataFrame
    ) -> Tuple[StandardScaler, KMeans, IsolationForest, float]:
        """Fit StandardScaler, KMeans, and IsolationForest on the dataset."""
        features = [f for f in MODELING_FEATURES if f in df.columns]
        X = df[features].values.astype(float)

        # Fill any remaining NaNs with column median
        for i in range(X.shape[1]):
            col_median = np.nanmedian(X[:, i])
            nan_mask = np.isnan(X[:, i])
            X[nan_mask, i] = col_median

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-Means
        kmeans = KMeans(
            n_clusters=3, random_state=RANDOM_SEED, n_init=10, max_iter=300
        )
        cluster_ids = kmeans.fit_predict(X_scaled)
        df["Cluster_ID"] = cluster_ids

        # Contamination rate from loneliness distribution
        if "loneliness_score" in df.columns:
            high_loneliness = (df["loneliness_score"] >= 4).sum()
            contamination = float(np.clip(high_loneliness / len(df), 0.01, 0.3))
        else:
            contamination = 0.05

        # Isolation Forest
        iso_forest = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=RANDOM_SEED,
            max_samples="auto",
        )
        preds = iso_forest.fit_predict(X_scaled)
        df["Anomaly_Flag"] = preds
        df["Anomaly_Score"] = iso_forest.decision_function(X_scaled)

        return scaler, kmeans, iso_forest, contamination

    # ------------------------------------------------------------------
    # Phase 4 helper  –  cluster-label mapping from centroids
    # ------------------------------------------------------------------
    def _derive_cluster_label_map(self) -> Dict[int, str]:
        """
        Assign semantic labels to KMeans clusters using centroid
        Social_Density_Score (same logic as pipeline Phase 3
        ``_assign_cluster_labels``).
        """
        features = [f for f in MODELING_FEATURES if f in self.df.columns]
        sds_idx = features.index("Social_Density_Score")

        centroids = self.kmeans.cluster_centers_
        profiles = {
            i: {"social_density": centroids[i][sds_idx]}
            for i in range(len(centroids))
        }

        sorted_clusters = sorted(profiles.items(), key=lambda x: x[1]["social_density"])

        label_map: Dict[int, str] = {}
        label_map[sorted_clusters[0][0]] = "At-Risk"
        label_map[sorted_clusters[1][0]] = "Academically Focused"
        label_map[sorted_clusters[-1][0]] = "Highly Social"

        logger.info("Cluster label map: %s", label_map)
        return label_map

    # ------------------------------------------------------------------
    # Campus baselines (for radar chart)
    # ------------------------------------------------------------------
    def _compute_campus_baselines(self) -> Dict[str, float]:
        """
        Compute campus-wide mean values for the six radar-chart
        dimensions, normalised to a 0-100 scale.
        """
        df = self.df

        def _safe_mean(col: str) -> float:
            if col in df.columns:
                return float(df[col].mean())
            return 0.0

        # Raw means
        sds_mean = _safe_mean("Social_Density_Score")
        study_mean = _safe_mean("daily_self_study_hours")
        screen_mean = _safe_mean("screen_time")
        wellness_mean = _safe_mean("social_wellness_rating")
        sleep_mean = _safe_mean("sleep_hours")
        social_activity_mean = (
            _safe_mean("meals_with_friends")
            + _safe_mean("social_outings")
            + _safe_mean("society_events_attended")
        ) / 3.0

        # Normalise to 0-100
        # SDS is typically in [0, ~0.5] — use dataset max for scaling
        sds_max = max(float(df["Social_Density_Score"].max()), 0.01)
        study_max = max(float(df["daily_self_study_hours"].max()), 0.01)
        screen_max = max(float(df["screen_time"].max()), 0.01)
        sleep_max = max(float(df["sleep_hours"].max()), 0.01)
        social_act_max = max(
            (
                float(df["meals_with_friends"].max())
                + float(df["social_outings"].max())
                + float(df["society_events_attended"].max())
            )
            / 3.0,
            0.01,
        )

        return {
            "Social Density": round(min((sds_mean / sds_max) * 100, 100), 1),
            "Academic Focus": round(min((study_mean / study_max) * 100, 100), 1),
            "Screen Time": round(min((screen_mean / screen_max) * 100, 100), 1),
            "Wellbeing": round(min((wellness_mean / 5.0) * 100, 100), 1),
            "Sleep Quality": round(min((sleep_mean / sleep_max) * 100, 100), 1),
            "Social Activity": round(
                min((social_activity_mean / social_act_max) * 100, 100), 1
            ),
        }

    # ------------------------------------------------------------------
    # Single-student inference
    # ------------------------------------------------------------------
    def predict(self, input_data: AssessmentInput) -> PredictionResponse:
        """
        Run the full inference pipeline for a single student.

        Steps mirror Phases 2-4 of the original pipeline but operate on
        a single feature vector rather than the whole dataset.
        """
        data = input_data.model_dump()

        # 1. Compute Social Density Score for this student ----------------
        communal_vals = np.array(
            [[data[c] for c in COMMUNAL_COLS]], dtype=float
        )
        isolation_vals = np.array([[data[ISOLATION_COL]]], dtype=float)

        communal_norm = self.communal_scaler.transform(communal_vals)
        isolation_norm = self.isolation_scaler.transform(isolation_vals)

        communal_mean = float(communal_norm.mean(axis=1)[0])
        isolation_flat = float(isolation_norm.flatten()[0])
        sds = communal_mean / (1.0 + isolation_flat)
        data["Social_Density_Score"] = sds

        # 2. Build feature vector in MODELING_FEATURES order --------------
        feature_vector = np.array(
            [[data.get(f, 0.0) for f in MODELING_FEATURES]], dtype=float
        )

        # 3. Scale --------------------------------------------------------
        X_scaled = self.standard_scaler.transform(feature_vector)

        # 4. Cluster prediction -------------------------------------------
        cluster_id = int(self.kmeans.predict(X_scaled)[0])
        cluster_label = self.cluster_label_map.get(cluster_id, "Unknown")

        # 5. Anomaly detection --------------------------------------------
        anomaly_pred = int(self.isolation_forest.predict(X_scaled)[0])
        anomaly_score = float(self.isolation_forest.decision_function(X_scaled)[0])
        is_anomaly = anomaly_pred == -1

        # 6. False-positive filter (Phase 4 logic) ------------------------
        risk_category = self._apply_false_positive_filter(
            data, sds, is_anomaly, cluster_label
        )

        # 7. Recommendation -----------------------------------------------
        recommendation = self._get_recommendation(risk_category)

        # 8. User scores for radar chart ----------------------------------
        user_scores = self._compute_user_scores(data)

        # 9. Build response -----------------------------------------------
        from privacy import generate_session_id

        return PredictionResponse(
            user_id=generate_session_id(),
            social_density_score=round(sds, 4),
            cluster_label=cluster_label,
            risk_category=risk_category,
            anomaly_score=round(anomaly_score, 4),
            recommendation=recommendation,
            user_scores=user_scores,
            campus_baseline=self.campus_baseline,
        )

    # ------------------------------------------------------------------
    # Phase 4 false-positive filter (single student)
    # ------------------------------------------------------------------
    def _apply_false_positive_filter(
        self,
        data: Dict[str, Any],
        sds: float,
        is_anomaly: bool,
        cluster_label: str,
    ) -> str:
        """
        Replicate the false-positive filter from Phase 4 for a single
        student.

        Priority order:
          1. High-academic exception → 'Academically Focused Overachiever'
          2. Genuine isolation confirmation → 'Confirmed At-Risk'
          3. Remaining anomalies → 'Monitor'
          4. Non-anomalies keep their cluster label.
        """
        if not is_anomaly:
            return cluster_label

        low_social = sds < self.social_density_median

        # Exception filter: high-performing introvert
        if (
            low_social
            and data.get("daily_self_study_hours", 0) >= self.study_75th
            and data.get("cgpa", 0) >= self.cgpa_threshold
            and data.get("class_attendance", 0) >= self.attendance_threshold
        ):
            return "Academically Focused Overachiever"

        # Genuine isolation: low social + high screen + high stress + low mood
        if (
            low_social
            and data.get("screen_time", 0) >= self.screen_time_75th
            and data.get("workload_stress", 0) >= self.stress_threshold
            and data.get("mood_rating", 10) <= self.mood_threshold
        ):
            return "Confirmed At-Risk"

        # If not low social but still an anomaly (e.g., extreme partying/screen time)
        if not low_social:
            return "Hyper-Social Anomaly"

        # Remaining anomaly that doesn't meet strict criteria
        return "Monitor"

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    @staticmethod
    def _get_recommendation(category: str) -> RecommendationPayload:
        """Return a context-appropriate recommendation payload."""
        if category in ("At-Risk", "Confirmed At-Risk", "Monitor"):
            return RecommendationPayload(
                status="warning",
                title="Let's Rebalance",
                message=(
                    "Your social-engagement signals are lower than the campus "
                    "average. Consider joining a casual meetup, study group, or "
                    "campus event this week — even a short coffee chat can make "
                    "a big difference!"
                ),
            )
        if category == "Hyper-Social Anomaly":
            return RecommendationPayload(
                status="warning",
                title="Realign Priorities",
                message=(
                    "Your social and extracurricular activities are extremely high, "
                    "which is great for networking, but it might be impacting your "
                    "academic focus or well-being. Consider dedicating a bit more "
                    "time to self-study and rest to maintain a healthy balance."
                ),
            )
        if category in ("Academically Focused Overachiever", "Academically Focused"):
            return RecommendationPayload(
                status="info",
                title="Strong Focus",
                message=(
                    "You're crushing it academically! Your study habits are "
                    "solid — just make sure to schedule some downtime with "
                    "friends. A library study-group session could combine the "
                    "best of both worlds."
                ),
            )
        # "Highly Social" or anything else
        return RecommendationPayload(
            status="success",
            title="Great Balance",
            message=(
                "You're maintaining a healthy social-academic balance — keep "
                "it up! Your engagement across campus activities is above "
                "average. Swing by the dining hall with a friend to keep the "
                "momentum going."
            ),
        )

    # ------------------------------------------------------------------
    # Radar-chart scores (single student)
    # ------------------------------------------------------------------
    def _compute_user_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Normalise the student's key metrics to 0-100 for radar-chart
        display, using the same maxima that were used for the campus
        baselines.
        """
        df = self.df

        sds = data.get("Social_Density_Score", 0.0)
        study = data.get("daily_self_study_hours", 0.0)
        screen = data.get("screen_time", 0.0)
        wellness = data.get("social_wellness_rating", 0.0)
        sleep = data.get("sleep_hours", 0.0)
        social_act = (
            data.get("meals_with_friends", 0.0)
            + data.get("social_outings", 0.0)
            + data.get("society_events_attended", 0.0)
        ) / 3.0

        sds_max = max(float(df["Social_Density_Score"].max()), 0.01)
        study_max = max(float(df["daily_self_study_hours"].max()), 0.01)
        screen_max = max(float(df["screen_time"].max()), 0.01)
        sleep_max = max(float(df["sleep_hours"].max()), 0.01)
        social_act_max = max(
            (
                float(df["meals_with_friends"].max())
                + float(df["social_outings"].max())
                + float(df["society_events_attended"].max())
            )
            / 3.0,
            0.01,
        )

        return {
            "Social Density": round(min((sds / sds_max) * 100, 100), 1),
            "Academic Focus": round(min((study / study_max) * 100, 100), 1),
            "Screen Time": round(min((screen / screen_max) * 100, 100), 1),
            "Wellbeing": round(min((wellness / 5.0) * 100, 100), 1),
            "Sleep Quality": round(min((sleep / sleep_max) * 100, 100), 1),
            "Social Activity": round(
                min((social_act / social_act_max) * 100, 100), 1
            ),
        }
