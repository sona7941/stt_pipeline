from __future__ import annotations

from contextlib import contextmanager
import os
import shutil
from pathlib import Path


def _ffmpeg_directory() -> Path:
    from imageio_ffmpeg import get_ffmpeg_exe

    ffmpeg_exe = Path(get_ffmpeg_exe())
    if not ffmpeg_exe.exists():
        raise FileNotFoundError("Bundled ffmpeg binary was not found.")

    cache_dir = Path(__file__).resolve().parents[2] / ".cache" / "ffmpeg"
    cache_dir.mkdir(parents=True, exist_ok=True)
    shim_path = cache_dir / "ffmpeg.exe"
    if not shim_path.exists():
        shutil.copy2(ffmpeg_exe, shim_path)
    return cache_dir


@contextmanager
def ffmpeg_on_path():
    directory = _ffmpeg_directory()
    previous_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{directory};{previous_path}" if previous_path else str(directory)
    try:
        yield
    finally:
        os.environ["PATH"] = previous_path
