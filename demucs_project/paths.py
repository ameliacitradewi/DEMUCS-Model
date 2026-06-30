from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


EXPECTED_FILES = ("mixture.wav", "drums.wav", "bass.wav", "other.wav", "vocals.wav")


def list_track_directories(dataset_root: Path, split: str) -> list[Path]:
    split_dir = dataset_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Folder split tidak ditemukan: {split_dir}")
    return sorted(path for path in split_dir.iterdir() if path.is_dir())


def find_track_directory(dataset_root: Path, split: str, track_name: str) -> Path:
    track_dir = dataset_root / split / track_name
    if not track_dir.is_dir():
        available = [path.name for path in list_track_directories(dataset_root, split)]
        preview = ", ".join(available[:8])
        raise FileNotFoundError(
            f"Track '{track_name}' tidak ditemukan pada split '{split}'. "
            f"Contoh track tersedia: {preview}"
        )
    return track_dir


def estimate_track_directory(
    separated_root: Path,
    model_name: str,
    track_name: str,
) -> Path:
    return separated_root / model_name / track_name


def ensure_expected_files(track_dir: Path, filenames: Iterable[str] = EXPECTED_FILES) -> None:
    missing = [name for name in filenames if not (track_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"File berikut tidak ditemukan di {track_dir}: {', '.join(missing)}"
        )


def safe_external_name(path: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", path.stem).strip(" ._")
    return cleaned or "external_track"
