from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audio import EPS, align_length, load_audio, match_channels, resample_audio
from .config import ProjectConfig
from .metrics import bss_eval_v4, evaluate_stem_metrics, finite_median
from .paths import estimate_track_directory, list_track_directories


def _prepare_estimate(
    estimate_path: Path,
    target_sample_rate: int,
    target_channels: int,
) -> tuple[np.ndarray, int]:
    estimate, estimate_sr = load_audio(estimate_path)
    estimate = match_channels(estimate, target_channels)
    estimate = resample_audio(estimate, estimate_sr, target_sample_rate)
    return estimate, estimate_sr


def evaluate_track(
    config: ProjectConfig,
    split: str,
    track_name: str,
    with_bss_eval: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference_dir = config.dataset_root / split / track_name
    estimate_dir = estimate_track_directory(
        config.separated_root,
        config.model.name,
        track_name,
    )
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Folder reference tidak ditemukan: {reference_dir}")
    if not estimate_dir.is_dir():
        raise FileNotFoundError(
            f"Folder estimate tidak ditemukan: {estimate_dir}. Jalankan inferensi dahulu."
        )

    mixture, sample_rate = load_audio(reference_dir / "mixture.wav")
    target_channels = mixture.shape[1]

    references: list[np.ndarray] = []
    estimates: list[np.ndarray] = []
    original_lengths: dict[str, dict[str, int]] = {}
    original_sample_rates: dict[str, int] = {}

    for stem in config.stems:
        reference, reference_sr = load_audio(reference_dir / f"{stem}.wav")
        if reference_sr != sample_rate:
            raise ValueError(
                f"Sample rate reference {stem} ({reference_sr}) berbeda dari mixture ({sample_rate})."
            )
        reference = match_channels(reference, target_channels)
        estimate, estimate_original_sr = _prepare_estimate(
            estimate_dir / f"{stem}.wav",
            target_sample_rate=sample_rate,
            target_channels=target_channels,
        )
        references.append(reference)
        estimates.append(estimate)
        original_lengths[stem] = {
            "reference": len(reference),
            "estimate_after_resample": len(estimate),
        }
        original_sample_rates[stem] = estimate_original_sr

    aligned, _ = align_length(mixture, *references, *estimates)
    mixture_aligned = aligned[0]
    source_count = len(config.stems)
    references_aligned = aligned[1 : 1 + source_count]
    estimates_aligned = aligned[1 + source_count :]
    aligned_samples = len(mixture_aligned)

    rows: list[dict[str, Any]] = []
    for stem, reference, estimate in zip(
        config.stems,
        references_aligned,
        estimates_aligned,
    ):
        row = evaluate_stem_metrics(
            reference=reference,
            estimate=estimate,
            mixture=mixture_aligned,
            n_fft=config.evaluation.n_fft,
            hop_length=config.evaluation.hop_length,
            top_db=config.evaluation.top_db,
        )
        row.update(
            {
                "split": split,
                "track": track_name,
                "stem": stem,
                "sample_rate": sample_rate,
                "channels": target_channels,
                "aligned_samples": aligned_samples,
                "duration_seconds": aligned_samples / sample_rate,
                "estimate_original_sample_rate": original_sample_rates[stem],
                "reference_original_samples": original_lengths[stem]["reference"],
                "estimate_samples_after_resample": original_lengths[stem][
                    "estimate_after_resample"
                ],
                "bss_eval_enabled": with_bss_eval,
            }
        )
        rows.append(row)

    reference_stack = np.stack(references_aligned, axis=0)
    estimate_stack = np.stack(estimates_aligned, axis=0)

    if with_bss_eval:
        bss = bss_eval_v4(
            references=reference_stack,
            estimates=estimate_stack,
            sample_rate=sample_rate,
            window_seconds=config.evaluation.bss_window_seconds,
            hop_seconds=config.evaluation.bss_hop_seconds,
            filter_length=config.evaluation.bss_filter_length,
        )
        for source_index, row in enumerate(rows):
            row["bss_sdr_db"] = finite_median(bss["sdr"][source_index])
            row["bss_isr_db"] = finite_median(bss["isr"][source_index])
            row["bss_sir_db"] = finite_median(bss["sir"][source_index])
            row["bss_sar_db"] = finite_median(bss["sar"][source_index])

    estimated_sum = np.sum(estimate_stack, axis=0)
    reference_sum = np.sum(reference_stack, axis=0)
    mixture_norm = np.linalg.norm(mixture_aligned) + EPS

    summary: dict[str, Any] = {
        "split": split,
        "track": track_name,
        "model": config.model.name,
        "sample_rate": sample_rate,
        "channels": target_channels,
        "aligned_samples": aligned_samples,
        "duration_seconds": aligned_samples / sample_rate,
        "estimate_mixture_consistency_error": float(
            np.linalg.norm(mixture_aligned - estimated_sum) / mixture_norm
        ),
        "reference_mixture_consistency_error": float(
            np.linalg.norm(mixture_aligned - reference_sum) / mixture_norm
        ),
        "median_si_sdr_db": float(np.nanmedian([row["si_sdr_db"] for row in rows])),
        "mean_si_sdr_db": float(np.nanmean([row["si_sdr_db"] for row in rows])),
        "median_si_sdri_db": float(np.nanmedian([row["si_sdri_db"] for row in rows])),
        "all_stems_positive_si_sdri": bool(
            all(row["si_sdri_db"] > 0 for row in rows)
        ),
        "bss_eval_enabled": with_bss_eval,
    }
    if with_bss_eval:
        summary["median_bss_sdr_db"] = float(
            np.nanmedian([row["bss_sdr_db"] for row in rows])
        )

    dataframe = pd.DataFrame(rows)
    output_dir = config.evaluation_root / config.model.name / split / track_name
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_dir / "stem_metrics.csv", index=False)
    with (output_dir / "track_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return dataframe, summary


def evaluate_dataset(
    config: ProjectConfig,
    split: str = "test",
    with_bss_eval: bool = False,
    limit: int | None = None,
    continue_on_error: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    track_dirs = list_track_directories(config.dataset_root, split)
    if limit is not None:
        track_dirs = track_dirs[:limit]

    metric_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for index, track_dir in enumerate(track_dirs, start=1):
        print(f"[{index}/{len(track_dirs)}] Evaluasi: {track_dir.name}")
        try:
            frame, summary = evaluate_track(
                config,
                split=split,
                track_name=track_dir.name,
                with_bss_eval=with_bss_eval,
            )
            metric_frames.append(frame)
            summaries.append(summary)
        except Exception as exc:
            errors.append(
                {
                    "split": split,
                    "track": track_dir.name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            if not continue_on_error:
                raise
            print(f"  Gagal: {type(exc).__name__}: {exc}")

    all_metrics = (
        pd.concat(metric_frames, ignore_index=True)
        if metric_frames
        else pd.DataFrame()
    )
    summary_frame = pd.DataFrame(summaries)
    error_frame = pd.DataFrame(errors)

    output_dir = config.evaluation_root / config.model.name / split
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(output_dir / "all_stem_metrics.csv", index=False)
    summary_frame.to_csv(output_dir / "all_track_summaries.csv", index=False)
    error_frame.to_csv(output_dir / "evaluation_errors.csv", index=False)

    return all_metrics, summary_frame, error_frame
