from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ProjectConfig


CORE_METRICS = [
    "si_sdr_db",
    "si_sdri_db",
    "waveform_sdr_db",
    "bss_sdr_db",
    "bss_sir_db",
    "bss_sar_db",
    "bss_isr_db",
    "spectral_convergence",
    "log_spectral_mae_db",
]


def _available_metrics(dataframe: pd.DataFrame) -> list[str]:
    return [metric for metric in CORE_METRICS if metric in dataframe.columns]


def _select_best_median_worst(track_scores: pd.DataFrame, metric: str) -> pd.DataFrame:
    valid = track_scores.dropna(subset=[metric]).sort_values(metric).reset_index(drop=True)
    if valid.empty:
        return pd.DataFrame()
    median_value = float(valid[metric].median())
    median_index = int((valid[metric] - median_value).abs().idxmin())
    selected = pd.concat(
        [
            valid.iloc[[0]].assign(case="worst"),
            valid.loc[[median_index]].assign(case="median"),
            valid.iloc[[-1]].assign(case="best"),
        ],
        ignore_index=True,
    )
    return selected[["case", "track", metric]]


def _plot_metric_boxplot(dataframe: pd.DataFrame, metric: str, output_path: Path) -> None:
    stems = list(dataframe["stem"].dropna().unique())
    values = [
        dataframe.loc[dataframe["stem"] == stem, metric].dropna().to_numpy()
        for stem in stems
    ]
    if not stems or not any(len(value) for value in values):
        return

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.boxplot(values, labels=stems, showmeans=True)
    axis.set_title(f"Distribusi {metric} per stem")
    axis.set_xlabel("Stem")
    axis.set_ylabel(metric)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_report(config: ProjectConfig, split: str = "test") -> dict[str, Path]:
    input_dir = config.evaluation_root / config.model.name / split
    metrics_path = input_dir / "all_stem_metrics.csv"
    track_summary_path = input_dir / "all_track_summaries.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Belum ada hasil evaluasi dataset: {metrics_path}. Jalankan evaluate-dataset."
        )

    metrics = pd.read_csv(metrics_path)
    track_summaries = (
        pd.read_csv(track_summary_path)
        if track_summary_path.is_file()
        else pd.DataFrame()
    )
    if metrics.empty:
        raise ValueError("File metrik kosong; tidak ada report yang dapat dibuat.")

    output_dir = config.reports_root / config.model.name / split
    output_dir.mkdir(parents=True, exist_ok=True)

    available = _available_metrics(metrics)
    aggregations = {}
    for metric in available:
        aggregations[f"{metric}_median"] = (metric, "median")
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")

    summary_by_stem = (
        metrics.groupby("stem", as_index=False)
        .agg(track_count=("track", "nunique"), **aggregations)
    )
    summary_by_stem.to_csv(output_dir / "summary_by_stem.csv", index=False)

    primary_metric = "bss_sdr_db" if "bss_sdr_db" in metrics.columns else "si_sdr_db"
    track_scores = (
        metrics.groupby("track", as_index=False)[primary_metric]
        .median()
        .sort_values(primary_metric)
    )
    track_scores.to_csv(output_dir / "track_scores.csv", index=False)

    cases = _select_best_median_worst(track_scores, primary_metric)
    cases.to_csv(output_dir / "best_median_worst.csv", index=False)

    for metric in available:
        _plot_metric_boxplot(
            metrics,
            metric,
            output_dir / f"boxplot_{metric}.png",
        )

    positive_fraction = float((metrics["si_sdri_db"] > 0).mean())
    median_overall = float(metrics[primary_metric].median())
    stem_lines = []
    for _, row in summary_by_stem.iterrows():
        stem_lines.append(
            f"- **{row['stem']}**: median {primary_metric} = "
            f"{row[f'{primary_metric}_median']:.3f} dB"
        )

    case_lines = []
    for _, row in cases.iterrows():
        case_lines.append(
            f"- {row['case']}: `{row['track']}` ({row[primary_metric]:.3f} dB)"
        )

    consistency_text = "Tidak tersedia."
    if not track_summaries.empty and "estimate_mixture_consistency_error" in track_summaries:
        consistency_text = (
            f"Median relative error = "
            f"{track_summaries['estimate_mixture_consistency_error'].median():.6f}."
        )

    findings = f"""# Ringkasan Otomatis Eksperimen Demucs V4

## Konfigurasi

- Model: `{config.model.name}`
- Split: `{split}`
- Jumlah track berhasil dievaluasi: {metrics['track'].nunique()}
- Metrik utama report: `{primary_metric}`

## Hasil agregat

Median keseluruhan `{primary_metric}` adalah **{median_overall:.3f} dB**.
Sebanyak **{positive_fraction * 100:.1f}%** kombinasi track-stem memiliki SI-SDR improvement positif terhadap mixture.

{chr(10).join(stem_lines)}

## Best, median, dan worst case

{chr(10).join(case_lines)}

## Mixture consistency

{consistency_text}

## Interpretasi yang diperbolehkan

Hasil ini mengukur performa pada split `{split}` yang memiliki ground-truth stems. Model dapat disebut layak sebagai prototype apabila mayoritas SI-SDRi positif, skor antar-stem tidak menunjukkan kegagalan sistematis, artefak pada visualisasi/listening test dapat diterima, dan kebutuhan komputasinya sesuai.

Hasil MUSDB18-HQ **belum membuktikan** bahwa model selalu akurat pada seluruh lagu pop eksternal. Untuk klaim generalisasi, tambahkan lagu eksternal dengan ground truth atau listening test terstruktur bila stems tidak tersedia. Perhatikan pula bahwa `other` adalah gabungan semua sumber selain drums, bass, dan vocals.
"""
    findings_path = output_dir / "automatic_findings.md"
    findings_path.write_text(findings, encoding="utf-8")

    return {
        "summary_by_stem": output_dir / "summary_by_stem.csv",
        "track_scores": output_dir / "track_scores.csv",
        "best_median_worst": output_dir / "best_median_worst.csv",
        "automatic_findings": findings_path,
    }
