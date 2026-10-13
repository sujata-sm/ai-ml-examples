from pydantic import BaseModel, Field
from enum import Enum


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")


class BatchTextInput(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")


class SentimentResult(BaseModel):
    text: str
    sentiment: SentimentLabel
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    scores: dict[str, float] = Field(..., description="Scores for each sentiment class")


class BatchSentimentResult(BaseModel):
    results: list[SentimentResult]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
