"""Pydantic response models for the API (API-SPEC.md section 4)."""
from pydantic import BaseModel


class PredictResponse(BaseModel):
    prediction: str
    probability: float
    model: str


class OverallMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


class SkinToneMetric(BaseModel):
    accuracy: float
    count: int


class MetricsResponse(BaseModel):
    overall: OverallMetrics
    confusion_matrix: list[list[int]]
    skin_tone: dict[str, SkinToneMetric]
    fairness_gap: float


class HealthResponse(BaseModel):
    status: str
