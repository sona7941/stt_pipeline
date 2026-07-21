from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    source_path: Path
    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str | None = None


@dataclass(frozen=True)
class SalesCallScore:
    score: int
    verdict: str
    reasons: list[str]
    recommendations: list[str]


@dataclass(frozen=True)
class SentimentAnalysis:
    compound: float
    positive: float
    neutral: float
    negative: float
    label: str
    summary: str


@dataclass(frozen=True)
class CallAnalysis:
    transcript: TranscriptResult
    score: SalesCallScore
    sentiment: SentimentAnalysis
    transcript_txt_path: Path
    transcript_json_path: Path
    rank: int = 0
    batch_decision: str = "review"
    batch_reason: str = ""


@dataclass(frozen=True)
class PipelineResult:
    calls: list[CallAnalysis]
    summary_path: Path
    markdown_path: Path
    json_path: Path
    shortlist_path: Path
    shortlist_csv_path: Path
    sentiment_report_path: Path
