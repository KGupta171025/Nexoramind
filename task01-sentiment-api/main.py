import os
import sys
from pathlib import Path

# Add current folder to sys.path to allow flexible execution
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import settings
from app.model import SentimentClassifier
from app.router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load the model on startup
    and clean up resources on shutdown.
    """
    logger.info("Starting Sentiment Analysis API - loading model weights...")
    app.state.classifier = SentimentClassifier(model_name=settings.MODEL_NAME)
    logger.info("Model loaded and ready for incoming inference requests.")
    yield
    logger.info("Shutting down Sentiment Analysis API...")
    if hasattr(app.state, "classifier"):
        del app.state.classifier


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Enable CORS for external frontend or dashboard consumption
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Clean and structured validation error response."""
    errors = []
    for err in exc.errors():
        loc = " -> ".join([str(x) for x in err.get("loc", [])])
        msg = err.get("msg", "Validation error")
        errors.append({"field": loc, "message": msg})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Input validation failed. Please check your request payload.",
            "errors": errors,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Graceful handling of unhandled exceptions."""
    logger.error(f"Unhandled exception during request: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred while processing the request.",
            "message": str(exc),
        },
    )


# Include API endpoints
app.include_router(router)


@app.get("/", tags=["System"])
async def root():
    """Root endpoint welcoming users and providing direct links to docs."""
    return {
        "message": f"Welcome to {settings.APP_NAME} v{settings.APP_VERSION}",
        "documentation": "/docs",
        "redoc": "/redoc",
        "health_check": "/api/health",
        "endpoints": {
            "single_analysis": "POST /api/analyze",
            "batch_analysis": "POST /api/analyze/batch (up to 10)",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)