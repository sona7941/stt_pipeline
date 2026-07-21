from __future__ import annotations

from dataclasses import dataclass, field

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .models import SentimentAnalysis, TranscriptResult


@dataclass(frozen=True)
class SentimentAnalyzer:
    analyzer: SentimentIntensityAnalyzer = field(default_factory=SentimentIntensityAnalyzer)

    def analyze(self, transcript: TranscriptResult) -> SentimentAnalysis:
        text = transcript.text.strip()
        if not text:
            return SentimentAnalysis(
                compound=0.0,
                positive=0.0,
                neutral=1.0,
                negative=0.0,
                label="neutral",
                summary="No spoken content was detected, so the sentiment is neutral by default.",
            )

        scores = self.analyzer.polarity_scores(text)
        compound = float(scores["compound"])
        if compound >= 0.05:
            label = "positive"
            summary = "The call transcript carries an overall positive tone."
        elif compound <= -0.05:
            label = "negative"
            summary = "The call transcript carries an overall negative tone."
        else:
            label = "neutral"
            summary = "The call transcript is balanced or emotionally neutral overall."

        return SentimentAnalysis(
            compound=compound,
            positive=float(scores["pos"]),
            neutral=float(scores["neu"]),
            negative=float(scores["neg"]),
            label=label,
            summary=summary,
        )