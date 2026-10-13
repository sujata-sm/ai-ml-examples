# Sentiment Analysis API

A FastAPI-based REST API for analyzing text sentiment using a pre-trained DistilBERT transformer model.

## Features

- Single text sentiment analysis
- Batch processing (up to 100 texts)
- Confidence scores for predictions
- Pre-trained transformer model (DistilBERT)
- Auto-generated API documentation

## Tech Stack

- **FastAPI** - Modern Python web framework
- **Transformers** - HuggingFace transformers library
- **DistilBERT** - Lightweight BERT model fine-tuned on SST-2

## Installation

```bash
cd nlp/sentiment_analysis_api
pip install -r requirements.txt
```

## Running the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check and model status |
| POST | `/analyze` | Analyze single text |
| POST | `/analyze/batch` | Analyze multiple texts |

## Usage Examples

### Analyze Single Text

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product! It works great."}'
```

Response:
```json
{
  "text": "I love this product! It works great.",
  "sentiment": "positive",
  "confidence": 0.9998,
  "scores": {
    "positive": 0.9998,
    "negative": 0.0002
  }
}
```

### Batch Analysis

```bash
curl -X POST "http://127.0.0.1:8000/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great service!", "Terrible experience.", "It was okay."]}'
```

## Interactive Docs

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Model Info

Uses `distilbert-base-uncased-finetuned-sst-2-english` from HuggingFace:
- Trained on Stanford Sentiment Treebank (SST-2)
- Binary classification: Positive / Negative
- Lightweight and fast inference
