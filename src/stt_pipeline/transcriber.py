from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import whisper

from .ffmpeg_utils import ffmpeg_on_path
from .models import TranscriptResult, TranscriptSegment


@dataclass(frozen=True)
class TranscriptionConfig:
    model_name: str = "base"
    language: str | None = None
    device: str | None = None


class WhisperTranscriber:
    def __init__(self, config: TranscriptionConfig | None = None) -> None:
        self.config = config or TranscriptionConfig()
        self._model = whisper.load_model(self.config.model_name, device=self.config.device)

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        with ffmpeg_on_path():
            result = self._model.transcribe(
                str(audio_path),
                language=self.config.language,
                fp16=False,
                condition_on_previous_text=False,
                verbose=False,
            )

        segments = [
            TranscriptSegment(start=float(segment["start"]), end=float(segment["end"]), text=segment["text"].strip())
            for segment in result.get("segments", [])
        ]
        text = " ".join(segment.text for segment in segments).strip() or result.get("text", "").strip()

        return TranscriptResult(
            source_path=audio_path,
            text=text,
            segments=segments,
            language=result.get("language"),
        )
