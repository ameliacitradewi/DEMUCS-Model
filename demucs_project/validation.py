from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from .audio import EPS, load_audio
from .config import ProjectConfig
from .paths import EXPECTED_FILES, list_track_directories


def _metadata_row(split: str, track_dir: Path) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    infos: dict[str, sf.SoundFile] = {}

    for filename in EXPECTED_FILES:
        path = track_dir / filename
        if not path.is_file():
            issues.append(f"missing:{filename}")
            continue
        infos[filename] = sf.info(path)

    sample_rates = {info.samplerate for info in infos.values()}
    channels = {info.channels for info in infos.values()}
    frame_counts = {info.frames for info in infos.values()}

    if len(sample_rates) > 1:
        issues.append("sample_rate_mismatch")
    if len(channels) > 1:
        issues.append("channel_mismatch")
    if len(frame_counts) > 1:
        issues.append("frame_count_mismatch")
    if sample_rates and sample_rates != {44100}:
        issues.append("sample_rate_not_44100")
    if channels and channels != {2}:
        issues.append("not_stereo")

    mixture = infos.get("mixture.wav")
    return (
        {
            "split": split,
            "track": track_dir.name,
            "files_found": len(infos),
            "sample_rate": mixture.samplerate if mixture else None,
            "channels": mixture.channels if mixture else None,
            "frames": mixture.frames if mixture else None,
            "duration_seconds": (
                mixture.frames / mixture.samplerate if mixture else None
            ),
            "status": "ok" if not issues else "issue",
            "issues": ";".join(issues),
        },
        issues,
    )


def mixture_sum_error(track_dir: Path) -> float:
    mixture, sample_rate = load_audio(track_dir / "mixture.wav")
    sources = []
    for stem in ("drums", "bass", "other", "vocals"):
        audio, stem_sr = load_audio(track_dir / f"{stem}.wav")
        if stem_sr != sample_rate:
            raise ValueError(f"Sample rate mismatch pada {track_dir.name}/{stem}.wav")
        sources.append(audio)

    minimum = min([len(mixture), *[len(source) for source in sources]])
    mixture = mixture[:minimum]
    source_sum = np.sum([source[:minimum] for source in sources], axis=0)
    return float(
        np.linalg.norm(mixture - source_sum) / (np.linalg.norm(mixture) + EPS)
    )


def validate_dataset(
    config: ProjectConfig,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    check_mixture_sum: bool = False,
    sum_check_limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names_by_split: dict[str, set[str]] = {}

    for split in splits:
        track_dirs = list_track_directories(config.dataset_root, split)
        names_by_split[split] = {path.name for path in track_dirs}
        for index, track_dir in enumerate(track_dirs):
            row, _ = _metadata_row(split, track_dir)
            if check_mixture_sum and (
                sum_check_limit is None or index < sum_check_limit
            ):
                try:
                    row["mixture_sum_relative_error"] = mixture_sum_error(track_dir)
                except Exception as exc:  # dicatat agar validasi track lain tetap lanjut
                    row["mixture_sum_relative_error"] = np.nan
                    suffix = f"mixture_sum_error:{type(exc).__name__}"
                    row["issues"] = ";".join(filter(None, [row["issues"], suffix]))
                    row["status"] = "issue"
            rows.append(row)

    overlaps: dict[str, list[str]] = {}
    split_list = list(splits)
    for index, left in enumerate(split_list):
        for right in split_list[index + 1 :]:
            overlap = sorted(names_by_split[left] & names_by_split[right])
            if overlap:
                overlaps[f"{left}_vs_{right}"] = overlap

    dataframe = pd.DataFrame(rows)
    summary = {
        "dataset_root": str(config.dataset_root),
        "split_counts": {
            split: int((dataframe["split"] == split).sum()) for split in splits
        },
        "tracks_with_issues": int((dataframe["status"] != "ok").sum()),
        "name_overlaps": overlaps,
        "note": (
            "MUSDB18-HQ resmi berisi 100 train dan 50 test. Folder valid dapat berasal "
            "dari pembagian internal train selama tidak tumpang tindih dengan test."
        ),
    }

    output_dir = config.reports_root / "dataset_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_dir / "validation_tracks.csv", index=False)
    with (output_dir / "validation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return dataframe, summary
