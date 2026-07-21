# Call STT Pipeline

This project transcribes audio files in `audio/` with Whisper and produces a candidate shortlist recommendation from each interview-style recording.

## What it does

- Reads MP4, WAV, M4A, MP3, AAC, FLAC, and WEBM audio files.
- Produces a transcript for each recording.
- Generates a shortlist recommendation with short reasons and next-step guidance.
- Writes machine-readable JSON and human-readable text/markdown outputs.
- Produces a sentiment report PDF for every processed batch.
- Exports the shortlist table as CSV for download.

## Run it

Use the Python executable inside the provided virtual environment:

```powershell
& 'c:\Users\Sona mathew\Desktop\Project_1\venv\Scripts\python.exe' run_pipeline.py --input audio --output output --model base
```

If you want stronger accuracy and can wait longer, try `--model small`.

## Deploy on Streamlit Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the GitHub repo.
3. Set the main file path to `app.py`.
4. Keep `runtime.txt`, `packages.txt`, and `.streamlit/config.toml` in the repo root so the app has the right Python version and ffmpeg support.
5. Deploy the app.

The app is designed to run from the `audio/` folder by default, and users can also upload their own recordings from the browser UI.

## Output

- `output/summary.json` contains the full structured result.
- `output/summary.md` is a readable report.
- `output/shortlist.json` contains only the recommended shortlist.
- `output/shortlist.csv` contains the candidate name, score, decision, and sentiment.
- `output/sentiment_report.pdf` contains the call-level sentiment report.
- `output/transcripts/*.txt` contains one transcript per call.
- `output/transcripts/*.json` contains transcript metadata and scoring details.
