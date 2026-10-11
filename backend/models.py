"""
Pydantic schemas for the Campus AI Backend.

Defines request/response models for the social isolation prediction API.
"""

from pydantic import BaseModel, Field
from typing import Dict, Literal


class AssessmentInput(BaseModel):
    """
    Input model representing all fields from the student assessment form.
    Each field maps directly to a column used in the ML pipeline.
    """

    living_status: Literal["Hostelite", "Day Scholar"] = Field(
        ...,
        description="Student living arrangement: 'Hostelite' or 'Day Scholar'",
    )
    friend_group_size: float = Field(
        ..., ge=0, le=40, description="Number of friends in the student's group"
    )
    friend_influence: float = Field(
        ...,
        ge=1,
        le=5,
        description="How much friends influence event/class attendance (1-5 scale)",
    )
    cgpa: float = Field(
        ..., ge=0, le=4.0, description="Current CGPA on a 4.0 scale"
    )
    class_attendance: float = Field(
        ...,
        ge=1,
        le=5,
        description="Average weekly class attendance rate (1-5 scale)",
    )
    daily_self_study_hours: float = Field(
        ..., ge=0, le=24, description="Average daily self-study hours"
    )
    society_memberships: float = Field(
        ..., ge=0, le=20, description="Number of active society memberships"
    )
    society_events_attended: float = Field(
        ...,
        ge=0,
        description="Number of society meetings/events attended this semester",
    )
    meals_with_friends: float = Field(
        ...,
        ge=0,
        description="Number of meals eaten with friends per week on average",
    )
    hours_alone_weekly: float = Field(
        ...,
        ge=0,
        description="Total hours spent alone per week on average",
    )
    common_room_frequency: float = Field(
        ...,
        ge=1,
        le=5,
        description="Frequency of using common rooms/lounges/library (1-5 scale)",
    )
    social_outings: float = Field(
        ...,
        ge=0,
        description="Times leaving campus for social reasons this week",
    )
    assignments_due: float = Field(
        ...,
        ge=0,
        description="Number of assignments/quizzes due this week",
    )
    workload_stress: float = Field(
        ...,
        ge=1,
        le=10,
        description="Perceived academic workload stress (1-10 scale)",
    )
    workload_management: float = Field(
        ...,
        ge=1,
        le=5,
        description="How well workload was managed this week (1-5 scale)",
    )
    screen_time: float = Field(
        ...,
        ge=0,
        description="Average daily phone screen time in hours (weekdays)",
    )
    sleep_hours: float = Field(
        ..., ge=0, le=24, description="Average hours of sleep per night (weekdays)"
    )
    social_wellness_rating: float = Field(
        ...,
        ge=1,
        le=5,
        description="Self-rated social interaction and wellness (1-5 scale)",
    )
    mood_rating: float = Field(
        ...,
        ge=1,
        le=10,
        description="Overall mood rating for the week (1-10 scale)",
    )
    loneliness_score: float = Field(
        ...,
        ge=1,
        le=5,
        description="Frequency of feeling socially isolated/lonely (1-5 scale)",
    )
    peer_satisfaction: float = Field(
        ...,
        ge=1,
        le=10,
        description="Satisfaction with peer interactions (1-10 scale)",
    )
    peer_helpfulness: float = Field(
        ...,
        ge=1,
        le=5,
        description="Helpfulness of peers in academic matters (1-5 scale)",
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "living_status": "Hostelite",
                    "friend_group_size": 5,
                    "friend_influence": 3,
                    "cgpa": 3.2,
                    "class_attendance": 4,
                    "daily_self_study_hours": 3,
                    "society_memberships": 2,
                    "society_events_attended": 4,
                    "meals_with_friends": 7,
                    "hours_alone_weekly": 15,
                    "common_room_frequency": 3,
                    "social_outings": 2,
                    "assignments_due": 3,
                    "workload_stress": 5,
                    "workload_management": 3,
                    "screen_time": 4,
                    "sleep_hours": 7,
                    "social_wellness_rating": 3,
                    "mood_rating": 6,
                    "loneliness_score": 2,
                    "peer_satisfaction": 7,
                    "peer_helpfulness": 3,
                }
            ]
        }
    }


class RecommendationPayload(BaseModel):
    """Actionable recommendation returned to the student."""

    status: str = Field(
        ..., description="Alert level: 'warning', 'info', or 'success'"
    )
    title: str = Field(..., description="Short headline for the recommendation")
    message: str = Field(..., description="Detailed recommendation text")


class PredictionResponse(BaseModel):
    """Full prediction response returned by the /predict endpoint."""

    user_id: str = Field(..., description="Anonymous session identifier")
    social_density_score: float = Field(
        ..., description="Computed Social Density Score (higher = more social)"
    )
    cluster_label: str = Field(
        ...,
        description="Behavioral cluster: 'Highly Social', 'Academically Focused', or 'At-Risk'",
    )
    risk_category: str = Field(
        ...,
        description="Final risk category after false-positive filtering",
    )
    anomaly_score: float = Field(
        ...,
        description="Isolation Forest anomaly score (lower = more anomalous)",
    )
    recommendation: RecommendationPayload = Field(
        ..., description="Actionable recommendation for the student"
    )
    user_scores: Dict[str, float] = Field(
        ...,
        description="Normalized scores (0-100) for radar chart display",
    )
    campus_baseline: Dict[str, float] = Field(
        ...,
        description="Campus-wide average scores (0-100) for radar chart comparison",
    )
