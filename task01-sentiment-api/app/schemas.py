from typing import List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    """Request schema for single text sentiment analysis."""
    text: str = Field(
        ...,
        description="The input text to classify.",
        min_length=1,
        max_length=2000,
        examples=["This product exceeded all my expectations, absolutely wonderful!"]
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Text cannot be empty or whitespace only.")
        return cleaned


class BatchRequest(BaseModel):
    """Request schema for batch sentiment analysis (up to 10 texts)."""
    texts: List[str] = Field(
        ...,
        description="List of text strings to classify (maximum 10 items).",
        min_length=1,
        max_length=10,
        examples=[
            [
                "I love using this API, it is exceptionally fast!",
                "The delivery arrived late and the packaging was damaged.",
                "The order arrived on Tuesday as scheduled."
            ]
        ]
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("Batch cannot be empty.")
        if len(value) > 10:
            raise ValueError("Batch size exceeds maximum limit of 10 items.")

        cleaned_list: List[str] = []
        for idx, text in enumerate(value):
            if not isinstance(text, str):
                raise ValueError(f"Item at index {idx} must be a string.")
            cleaned = text.strip()
            if not cleaned:
                raise ValueError(f"Item at index {idx} cannot be empty or whitespace only.")
            if len(cleaned) > 2000:
                raise ValueError(f"Item at index {idx} exceeds maximum length of 2000 characters.")
            cleaned_list.append(cleaned)
        return cleaned_list


class ClassScores(BaseModel):
    """Per-class probability scores for positive, negative, and neutral."""
    positive: float = Field(..., description="Probability of positive sentiment (0.0 to 1.0).", ge=0.0, le=1.0)
    negative: float = Field(..., description="Probability of negative sentiment (0.0 to 1.0).", ge=0.0, le=1.0)
    neutral: float = Field(..., description="Probability of neutral sentiment (0.0 to 1.0).", ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    """Response schema for single sentiment analysis."""
    sentiment: str = Field(..., description="Predicted sentiment label: 'positive', 'negative', or 'neutral'.")
    confidence: float = Field(..., description="Confidence score of the winning class (0.0 to 1.0).")
    scores: ClassScores = Field(..., description="Individual probability scores for all classes.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sentiment": "positive",
                "confidence": 0.9873,
                "scores": {
                    "positive": 0.9873,
                    "negative": 0.0051,
                    "neutral": 0.0076
                }
            }
        }
    }


class BatchResponse(BaseModel):
    """Response schema for batch sentiment analysis."""
    results: List[AnalyzeResponse] = Field(..., description="List of sentiment analysis results.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "results": [
                    {
                        "sentiment": "positive",
                        "confidence": 0.9873,
                        "scores": {
                            "positive": 0.9873,
                            "negative": 0.0051,
                            "neutral": 0.0076
                        }
                    },
                    {
                        "sentiment": "negative",
                        "confidence": 0.9421,
                        "scores": {
                            "positive": 0.0125,
                            "negative": 0.9421,
                            "neutral": 0.0454
                        }
                    },
                    {
                        "sentiment": "neutral",
                        "confidence": 0.8814,
                        "scores": {
                            "positive": 0.0342,
                            "negative": 0.0844,
                            "neutral": 0.8814
                        }
                    }
                ]
            }
        }
    }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field("healthy", description="API health status.")
    model: str = Field(..., description="Name of the underlying Hugging Face model.")
    version: str = Field(..., description="API Version.")


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    detail: str = Field(..., description="Human-readable description of the error.")
    errors: Optional[List[Any]] = Field(None, description="Detailed validation error traces if applicable.")
