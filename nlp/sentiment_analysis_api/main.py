from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from models import (
    TextInput,
    BatchTextInput,
    SentimentResult,
    BatchSentimentResult,
    HealthResponse,
)
from analyzer import analyzer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    print("Loading sentiment analysis model...")
    analyzer.load_model()
    print("Model loaded successfully!")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyze text sentiment using DistilBERT transformer model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Sentiment Analysis API",
        "docs": "/docs",
        "endpoints": {
            "analyze": "POST /analyze",
            "batch": "POST /analyze/batch",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=analyzer.is_loaded(),
        model_name=analyzer.model_name,
    )


@app.post("/analyze", response_model=SentimentResult, tags=["Sentiment"])
def analyze_text(input: TextInput):
    """
    Analyze sentiment of a single text.

    Returns sentiment (positive/negative/neutral) with confidence score.
    """
    try:
        return analyzer.analyze(input.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch", response_model=BatchSentimentResult, tags=["Sentiment"])
def analyze_batch(input: BatchTextInput):
    """
    Analyze sentiment of multiple texts.

    Accepts up to 100 texts in a single request.
    """
    try:
        results = analyzer.analyze_batch(input.texts)
        return BatchSentimentResult(results=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")
