from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    device: str
    shifts: int
    overlap: float
    jobs: int
    segment: int | None


@dataclass(frozen=True)
class EvaluationConfig:
    n_fft: int
    hop_length: int
    top_db: float
    bss_window_seconds: float
    bss_hop_seconds: float
    bss_filter_length: int


@dataclass(frozen=True)
class VisualizationConfig:
    start_seconds: float
    duration_seconds: float
    n_fft: int
    hop_length: int
    top_db: float


@dataclass(frozen=True)
class ProjectConfig:
    config_path: Path
    dataset_root: Path
    output_root: Path
    stems: tuple[str, ...]
    model: ModelConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig

    @property
    def separated_root(self) -> Path:
        return self.output_root / "separated"

    @property
    def evaluation_root(self) -> Path:
        return self.output_root / "evaluation"

    @property
    def figures_root(self) -> Path:
        return self.output_root / "figures"

    @property
    def reports_root(self) -> Path:
        return self.output_root / "reports"


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Konfigurasi wajib '{key}' tidak ditemukan.")
    return mapping[key]


def _resolve_path(raw: str | Path, base_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_config(path: str | Path = "config.yaml") -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    base_dir = config_path.parent
    model_raw = _required(raw, "model")
    eval_raw = _required(raw, "evaluation")
    vis_raw = _required(raw, "visualization")

    stems = tuple(raw.get("stems", ["drums", "bass", "other", "vocals"]))
    if set(stems) != {"drums", "bass", "other", "vocals"}:
        raise ValueError(
            "Untuk evaluasi MUSDB18-HQ empat-stem, 'stems' harus berisi "
            "drums, bass, other, dan vocals."
        )

    config = ProjectConfig(
        config_path=config_path,
        dataset_root=_resolve_path(_required(raw, "dataset_root"), base_dir),
        output_root=_resolve_path(raw.get("output_root", "outputs"), base_dir),
        stems=stems,
        model=ModelConfig(
            name=str(model_raw.get("name", "htdemucs_ft")),
            device=str(model_raw.get("device", "auto")),
            shifts=int(model_raw.get("shifts", 1)),
            overlap=float(model_raw.get("overlap", 0.25)),
            jobs=int(model_raw.get("jobs", 0)),
            segment=(
                None
                if model_raw.get("segment") is None
                else int(model_raw["segment"])
            ),
        ),
        evaluation=EvaluationConfig(
            n_fft=int(eval_raw.get("n_fft", 4096)),
            hop_length=int(eval_raw.get("hop_length", 1024)),
            top_db=float(eval_raw.get("top_db", 80)),
            bss_window_seconds=float(eval_raw.get("bss_window_seconds", 2.0)),
            bss_hop_seconds=float(eval_raw.get("bss_hop_seconds", 1.5)),
            bss_filter_length=int(eval_raw.get("bss_filter_length", 512)),
        ),
        visualization=VisualizationConfig(
            start_seconds=float(vis_raw.get("start_seconds", 30.0)),
            duration_seconds=float(vis_raw.get("duration_seconds", 10.0)),
            n_fft=int(vis_raw.get("n_fft", 4096)),
            hop_length=int(vis_raw.get("hop_length", 1024)),
            top_db=float(vis_raw.get("top_db", 80)),
        ),
    )

    if not 0.0 <= config.model.overlap < 1.0:
        raise ValueError("model.overlap harus berada pada rentang [0, 1).")
    if config.model.shifts < 0:
        raise ValueError("model.shifts tidak boleh negatif.")
    if config.model.segment is not None and config.model.segment <= 0:
        raise ValueError("model.segment harus positif atau null.")

    for directory in (
        config.output_root,
        config.separated_root,
        config.evaluation_root,
        config.figures_root,
        config.reports_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return config
