from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, process_audio_folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe interview recordings and score them for candidate shortlist recommendation.")
    parser.add_argument("--input", type=Path, default=Path("audio"), help="Folder containing audio recordings.")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Folder for transcripts and reports.")
    parser.add_argument("--model", default="base", help="Whisper model name, for example tiny, base, small, medium, or large.")
    parser.add_argument("--language", default=None, help="Optional ISO language code such as en.")
    parser.add_argument("--device", default=None, help="Optional Whisper device, such as cpu or cuda.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = PipelineConfig(model_name=args.model, language=args.language, device=args.device)
    result = process_audio_folder(args.input, args.output, config)

    shortlisted = sum(1 for call in result.calls if call.batch_decision == "recommended_shortlist")
    print(
        f"Processed {len(result.calls)} audio file(s). "
        f"Shortlisted {shortlisted}. Summary written to {result.summary_path}"
    )
    return 0
