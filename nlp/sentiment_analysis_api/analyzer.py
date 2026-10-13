from transformers import pipeline
from models import SentimentLabel, SentimentResult


class SentimentAnalyzer:
    """Sentiment analyzer using HuggingFace transformers."""

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name = model_name
        self.classifier = None

    def load_model(self):
        """Load the sentiment analysis model."""
        if self.classifier is None:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                top_k=None,  # Return all scores
            )
        return True

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.classifier is not None

    def _map_label(self, label: str) -> SentimentLabel:
        """Map model labels to our standard labels."""
        label = label.upper()
        if label in ["POSITIVE", "POS", "1", "LABEL_1"]:
            return SentimentLabel.POSITIVE
        elif label in ["NEGATIVE", "NEG", "0", "LABEL_0"]:
            return SentimentLabel.NEGATIVE
        return SentimentLabel.NEUTRAL

    def _parse_result(self, text: str, result: list[dict]) -> SentimentResult:
        """Parse a single result from the classifier."""
        scores = {}
        best_label = None
        best_score = 0.0

        for item in result:
            label = self._map_label(item["label"])
            score = item["score"]
            scores[label.value] = round(score, 4)

            if score > best_score:
                best_score = score
                best_label = label

        return SentimentResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            sentiment=best_label,
            confidence=round(best_score, 4),
            scores=scores,
        )

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text."""
        if not self.is_loaded():
            self.load_model()

        result = self.classifier(text)[0]
        return self._parse_result(text, result)

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment of multiple texts."""
        if not self.is_loaded():
            self.load_model()

        results = self.classifier(texts)
        return [
            self._parse_result(text, result)
            for text, result in zip(texts, results)
        ]


# Global analyzer instance
analyzer = SentimentAnalyzer()
