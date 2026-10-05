from fastapi import APIRouter, Request, HTTPException, status
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ErrorResponse,
)
from app.config import settings

router = APIRouter(prefix="/api", tags=["Sentiment Analysis"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze sentiment of a single text",
    description="Classifies the given text as positive, negative, or neutral and returns confidence and per-class probability scores.",
    responses={
        200: {"model": AnalyzeResponse, "description": "Successful sentiment analysis prediction"},
        422: {"model": ErrorResponse, "description": "Validation error (e.g. empty text or exceeds length limit)"},
        500: {"model": ErrorResponse, "description": "Internal model inference failure"},
    },
)
async def analyze_sentiment(payload: AnalyzeRequest, request: Request):
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment classification model is not loaded yet.",
        )

    try:
        results = classifier.predict([payload.text])
        return results[0]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )


@router.post(
    "/analyze/batch",
    response_model=BatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze sentiment for a batch of texts",
    description="Batch sentiment classification for up to 10 texts. Returns a list of predictions matching input order.",
    responses={
        200: {"model": BatchResponse, "description": "Successful batch sentiment classification"},
        422: {"model": ErrorResponse, "description": "Validation error (e.g. >10 texts or empty strings)"},
        500: {"model": ErrorResponse, "description": "Internal model inference failure"},
    },
)
async def analyze_sentiment_batch(payload: BatchRequest, request: Request):
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment classification model is not loaded yet.",
        )

    try:
        results = classifier.predict(payload.texts)
        return {"results": results}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(exc)}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service Health Check",
    description="Returns the operational status of the API and loaded model info.",
)
async def health_check():
    return {
        "status": "healthy",
        "model": settings.MODEL_NAME,
        "version": settings.APP_VERSION,
    }
