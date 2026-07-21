from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .models import CallAnalysis, PipelineResult
from .report import write_sentiment_report_pdf
from .sentiment import SentimentAnalyzer
from .scoring import CandidateShortlistScorer
from .transcriber import TranscriptionConfig, WhisperTranscriber


SUPPORTED_EXTENSIONS = {".mp4", ".m4a", ".wav", ".mp3", ".aac", ".flac", ".webm"}


@dataclass(frozen=True)
class PipelineConfig:
    model_name: str = "base"
    language: str | None = None
    device: str | None = None
    shortlist_ratio: float = 0.10
    review_ratio: float = 0.20


def process_audio_folder(input_dir: Path, output_dir: Path, config: PipelineConfig) -> PipelineResult:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir = output_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    transcriber = WhisperTranscriber(TranscriptionConfig(model_name=config.model_name, language=config.language, device=config.device))
    scorer = CandidateShortlistScorer()
    sentiment_analyzer = SentimentAnalyzer()

    calls: list[CallAnalysis] = []
    for audio_path in _audio_files(input_dir):
        transcript = transcriber.transcribe(audio_path)
        score = scorer.score(transcript)
        sentiment = sentiment_analyzer.analyze(transcript)

        safe_stem = audio_path.stem.replace(" ", "_")
        transcript_txt_path = transcript_dir / f"{safe_stem}.txt"
        transcript_json_path = transcript_dir / f"{safe_stem}.json"

        transcript_txt_path.write_text(transcript.text + "\n", encoding="utf-8")
        transcript_json_path.write_text(
            json.dumps(
                {
                    "source_path": str(audio_path),
                    "transcript": {
                        "text": transcript.text,
                        "language": transcript.language,
                        "segments": [asdict(segment) for segment in transcript.segments],
                    },
                    "score": asdict(score),
                    "sentiment": asdict(sentiment),
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )

        calls.append(
            CallAnalysis(
                transcript=transcript,
                score=score,
                sentiment=sentiment,
                transcript_txt_path=transcript_txt_path,
                transcript_json_path=transcript_json_path,
            )
        )

    ranked_calls = _rank_and_label_calls(calls, config.shortlist_ratio, config.review_ratio)

    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    shortlist_path = output_dir / "shortlist.json"
    shortlist_csv_path = output_dir / "shortlist.csv"
    sentiment_report_path = output_dir / "sentiment_report.pdf"

    summary_payload = _summary_payload(ranked_calls, input_dir, config)
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    markdown_path.write_text(_summary_markdown(ranked_calls, input_dir, config), encoding="utf-8")
    shortlist_path.write_text(json.dumps(_shortlist_payload(ranked_calls, input_dir, config), indent=2, ensure_ascii=True), encoding="utf-8")
    _write_shortlist_csv(shortlist_csv_path, ranked_calls)
    write_sentiment_report_pdf(sentiment_report_path, ranked_calls, input_dir, config.model_name)

    return PipelineResult(
        calls=ranked_calls,
        summary_path=summary_path,
        markdown_path=markdown_path,
        json_path=summary_path,
        shortlist_path=shortlist_path,
        shortlist_csv_path=shortlist_csv_path,
        sentiment_report_path=sentiment_report_path,
    )


def _audio_files(input_dir: Path) -> Iterable[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def _rank_and_label_calls(calls: list[CallAnalysis], shortlist_ratio: float, review_ratio: float) -> list[CallAnalysis]:
    if not calls:
        return []

    shortlist_count = max(1, int(round(len(calls) * shortlist_ratio)))
    review_count = max(0, int(round(len(calls) * review_ratio)))

    ranked_calls = sorted(calls, key=lambda call: (-call.score.score, call.transcript.source_path.name))
    labeled_calls: list[CallAnalysis] = []

    for index, call in enumerate(ranked_calls, start=1):
        if index <= shortlist_count:
            batch_decision = "recommended_shortlist"
            batch_reason = f"Rank {index} of {len(ranked_calls)} and within the top {shortlist_count} candidates."
        elif index <= shortlist_count + review_count:
            batch_decision = "needs_review"
            batch_reason = f"Rank {index} of {len(ranked_calls)} and within the review band after the shortlist cutoff."
        else:
            batch_decision = "do_not_shortlist"
            batch_reason = f"Rank {index} of {len(ranked_calls)} and below the shortlist and review bands."

        labeled_calls.append(
            CallAnalysis(
                transcript=call.transcript,
                score=call.score,
                sentiment=call.sentiment,
                transcript_txt_path=call.transcript_txt_path,
                transcript_json_path=call.transcript_json_path,
                rank=index,
                batch_decision=batch_decision,
                batch_reason=batch_reason,
            )
        )

    return labeled_calls


def _summary_payload(calls: list[CallAnalysis], input_dir: Path, config: PipelineConfig) -> dict[str, object]:
    return {
        "input_dir": str(input_dir),
        "model_name": config.model_name,
        "language": config.language,
        "device": config.device,
        "shortlist_ratio": config.shortlist_ratio,
        "review_ratio": config.review_ratio,
        "call_count": len(calls),
        "calls": [
            {
                "rank": call.rank,
                "batch_decision": call.batch_decision,
                "batch_reason": call.batch_reason,
                "source_path": str(call.transcript.source_path),
                "score": asdict(call.score),
                "sentiment": asdict(call.sentiment),
                "transcript_path": str(call.transcript_txt_path),
                "transcript_json_path": str(call.transcript_json_path),
            }
            for call in calls
        ],
    }


def _summary_markdown(calls: list[CallAnalysis], input_dir: Path, config: PipelineConfig) -> str:
    shortlist_count = max(1, int(round(len(calls) * config.shortlist_ratio)))
    review_count = max(0, int(round(len(calls) * config.review_ratio)))

    lines = [
        "# Call STT Summary",
        "",
        f"- Input folder: `{input_dir}`",
        f"- Whisper model: `{config.model_name}`",
        f"- Calls processed: {len(calls)}",
        f"- Recommended shortlist size: {shortlist_count}",
        f"- Review band size: {review_count}",
        "",
        "## Sentiment Overview",
        "",
        "| Rank | File | Sentiment | Compound | Score | Decision |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]

    for call in calls:
        lines.append(
            f"| {call.rank} | {call.transcript.source_path.name} | {call.sentiment.label} | {call.sentiment.compound:.2f} | {call.score.score} | {call.batch_decision} |"
        )

    lines.extend(["", "## Recommended Shortlist", "", "| Rank | File | Score | Decision |", "| ---: | --- | ---: | --- |"])

    for call in calls:
        if call.batch_decision == "recommended_shortlist":
            lines.append(f"| {call.rank} | {call.transcript.source_path.name} | {call.score.score} | {call.batch_decision} |")

    lines.extend(["", "## Review Queue", "", "| Rank | File | Score | Decision |", "| ---: | --- | ---: | --- |"])

    for call in calls:
        if call.batch_decision == "needs_review":
            lines.append(f"| {call.rank} | {call.transcript.source_path.name} | {call.score.score} | {call.batch_decision} |")

    lines.extend(["", "## Do Not Shortlist", "", "| Rank | File | Score | Decision |", "| ---: | --- | ---: | --- |"])

    for call in calls:
        if call.batch_decision == "do_not_shortlist":
            lines.append(f"| {call.rank} | {call.transcript.source_path.name} | {call.score.score} | {call.batch_decision} |")

    lines.extend(["", "## Notes", ""])
    for call in calls:
        lines.append(f"### {call.transcript.source_path.name}")
        lines.append(f"Rank: {call.rank}")
        lines.append(f"Score: {call.score.score} ({call.score.verdict})")
        lines.append(f"Sentiment: {call.sentiment.label} ({call.sentiment.compound:.2f})")
        lines.append(f"Sentiment summary: {call.sentiment.summary}")
        lines.append(f"Batch decision: {call.batch_decision}")
        if call.batch_reason:
            lines.append(f"Reason: {call.batch_reason}")
        if call.score.reasons:
            lines.append("Reasons:")
            for reason in call.score.reasons:
                lines.append(f"- {reason}")
        if call.score.recommendations:
            lines.append("Recommendations:")
            for recommendation in call.score.recommendations:
                lines.append(f"- {recommendation}")
        lines.append("")

    return "\n".join(lines)


def _shortlist_payload(calls: list[CallAnalysis], input_dir: Path, config: PipelineConfig) -> dict[str, object]:
    shortlist = [call for call in calls if call.batch_decision == "recommended_shortlist"]
    return {
        "input_dir": str(input_dir),
        "model_name": config.model_name,
        "shortlist_ratio": config.shortlist_ratio,
        "review_ratio": config.review_ratio,
        "shortlist_count": len(shortlist),
        "candidates": [
            {
                "rank": call.rank,
                "source_path": str(call.transcript.source_path),
                "score": call.score.score,
                "verdict": call.score.verdict,
                "batch_reason": call.batch_reason,
                "sentiment": asdict(call.sentiment),
                "transcript_path": str(call.transcript_txt_path),
            }
            for call in shortlist
        ],
    }


def _write_shortlist_csv(csv_path: Path, calls: list[CallAnalysis]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_name", "score", "decision", "sentiment", "compound_score"],
        )
        writer.writeheader()
        for call in calls:
            writer.writerow(
                {
                    "candidate_name": call.transcript.source_path.stem,
                    "score": call.score.score,
                    "decision": call.batch_decision,
                    "sentiment": call.sentiment.label,
                    "compound_score": f"{call.sentiment.compound:.2f}",
                }
            )
