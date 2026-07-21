from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stt_pipeline.pipeline import PipelineConfig, process_audio_folder

SUPPORTED_EXTENSIONS = {".mp4", ".m4a", ".wav", ".mp3", ".aac", ".flac", ".webm"}

st.set_page_config(page_title="Call Shortlisting App", page_icon="🎙️", layout="wide")
st.title("Call transcription and shortlist review")
st.caption("Upload new recordings or point to an existing folder, then review the ranked shortlist from the browser.")

with st.sidebar:
    st.header("Configuration")
    upload_folder = st.text_input("Input folder", value="audio")
    output_folder = st.text_input("Output folder", value="output")
    model_name = st.selectbox("Whisper model", ["base", "small", "medium", "large"], index=0)
    shortlist_ratio = st.slider("Shortlist ratio", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
    review_ratio = st.slider("Review ratio", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    uploaded_files = st.file_uploader(
        "Upload audio files",
        type=list(SUPPORTED_EXTENSIONS - {".mp4"}) + ["mp4"],
        accept_multiple_files=True,
    )
    run_button = st.button("Run processing", use_container_width=True)


@st.cache_data(show_spinner=False)
def discover_audio_files(folder: str) -> list[str]:
    folder_path = Path(folder)
    if not folder_path.exists():
        return []
    return sorted(str(path) for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def save_uploaded_files(files, target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []
    for uploaded in files or []:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        destination = target_dir / uploaded.name
        with destination.open("wb") as handle:
            handle.write(uploaded.getbuffer())
        saved_files.append(destination)
    return saved_files


input_dir = Path(upload_folder)
output_dir = Path(output_folder)

if uploaded_files:
    temp_root = ROOT / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="streamlit_uploads_", dir=str(temp_root)))
    saved_files = save_uploaded_files(uploaded_files, temp_dir)
    input_dir = temp_dir
    st.sidebar.success(f"Saved {len(saved_files)} uploaded file(s) to {temp_dir}")

if run_button:
    if not input_dir.exists() or not any(path.is_file() for path in input_dir.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS):
        st.error("No audio files were found. Add files in the input folder or upload some recordings.")
        st.stop()

    with st.spinner("Processing recordings..."):
        config = PipelineConfig(
            model_name=model_name,
            language=None,
            device=None,
            shortlist_ratio=float(shortlist_ratio),
            review_ratio=float(review_ratio),
        )
        result = process_audio_folder(input_dir, output_dir, config)

    st.success(
        f"Processed {len(result.calls)} call(s). Shortlisted {sum(1 for call in result.calls if call.batch_decision == 'recommended_shortlist')}"
    )

    shortlist_csv_path = result.shortlist_csv_path
    if shortlist_csv_path.exists():
        shortlist_csv_content = shortlist_csv_path.read_text(encoding="utf-8")
        st.download_button("Download shortlist CSV", shortlist_csv_content, file_name="shortlist.csv", mime="text/csv")

    sentiment_report_path = result.sentiment_report_path
    if sentiment_report_path.exists():
        st.download_button(
            "Download sentiment PDF",
            sentiment_report_path.read_bytes(),
            file_name="sentiment_report.pdf",
            mime="application/pdf",
        )

    st.subheader("Ranked shortlist")
    ranked_calls = sorted(result.calls, key=lambda call: (call.rank, call.transcript.source_path.name))
    table_data = [
        {
            "Candidate": Path(call.transcript.source_path).stem,
            "Score": call.score.score,
            "Decision": call.batch_decision,
            "Sentiment": call.sentiment.label,
            "Compound": f"{call.sentiment.compound:.2f}",
        }
        for call in ranked_calls
    ]
    st.dataframe(table_data, use_container_width=True, hide_index=True)

    if ranked_calls:
        selected_file = st.selectbox("Inspect a transcript", [Path(call.transcript.source_path).name for call in ranked_calls])
        selected_call = next(call for call in ranked_calls if Path(call.transcript.source_path).name == selected_file)
        st.subheader(selected_file)
        st.write(selected_call.transcript.text)
        with st.expander("Scoring details"):
            st.write(selected_call.score.reasons)
            st.write(selected_call.score.recommendations)
        with st.expander("Sentiment details"):
            st.write(selected_call.sentiment.summary)
            st.write(
                {
                    "label": selected_call.sentiment.label,
                    "compound": selected_call.sentiment.compound,
                    "positive": selected_call.sentiment.positive,
                    "neutral": selected_call.sentiment.neutral,
                    "negative": selected_call.sentiment.negative,
                }
            )

else:
    st.subheader("Detected files")
    detected_files = discover_audio_files(str(input_dir)) if input_dir.exists() else []
    if detected_files:
        st.write(detected_files)
    else:
        st.info("No audio files were found yet. Upload files or point the input folder to the recordings directory.")
