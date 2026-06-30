from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import soundfile as sf

from .config import ProjectConfig
from .paths import estimate_track_directory


_MODEL_CACHE: dict[str, Any] = {}


def _resolve_device(requested: str) -> str:
    if requested.lower() != "auto":
        return requested

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        from demucs.pretrained import get_model

        model = get_model(model_name)
        model.cpu()
        model.eval()
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


def _stems_complete(directory: Path, stems: tuple[str, ...]) -> bool:
    return all((directory / f"{stem}.wav").is_file() for stem in stems)


def _run_demucs_raw(
    config: ProjectConfig,
    input_audio: Path,
    destination: Path,
) -> dict[str, Any]:
    """Menjalankan core Demucs dan menyimpan float32 mentah tanpa rescale/clamp."""
    import torch
    from demucs.apply import apply_model
    from demucs.separate import load_track

    model_load_started = time.perf_counter()
    model_was_cached = config.model.name in _MODEL_CACHE
    model = _get_model(config.model.name)
    model_load_seconds = time.perf_counter() - model_load_started

    device = _resolve_device(config.model.device)
    wav = load_track(input_audio, model.audio_channels, model.samplerate)

    # Normalisasi ini mengikuti pipeline inferensi resmi Demucs 4.0.1.
    reference = wav.mean(0)
    reference_mean = reference.mean()
    reference_std = reference.std()
    normalized = wav - reference_mean
    normalized = normalized / (reference_std + 1e-8)

    inference_started = time.perf_counter()
    with torch.inference_mode():
        separated = apply_model(
            model,
            normalized[None],
            device=device,
            shifts=config.model.shifts,
            split=True,
            overlap=config.model.overlap,
            progress=True,
            num_workers=config.model.jobs,
            segment=config.model.segment,
        )[0]
    inference_seconds = time.perf_counter() - inference_started

    separated = separated * (reference_std + 1e-8)
    separated = separated + reference_mean

    destination.mkdir(parents=True, exist_ok=True)
    produced_stems = []
    for source, stem_name in zip(separated, model.sources):
        output_path = destination / f"{stem_name}.wav"
        # Demucs tensor: (channels, samples); soundfile: (samples, channels).
        audio = source.detach().cpu().numpy().T.astype("float32", copy=False)
        sf.write(output_path, audio, model.samplerate, subtype="FLOAT")
        produced_stems.append(stem_name)

    missing_expected = sorted(set(config.stems) - set(produced_stems))
    if missing_expected:
        raise RuntimeError(
            "Model tidak menghasilkan stem yang dibutuhkan: "
            + ", ".join(missing_expected)
        )

    return {
        "backend": "demucs_core_python",
        "resolved_device": device,
        "model_was_cached": model_was_cached,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "sample_rate": int(model.samplerate),
        "audio_channels": int(model.audio_channels),
        "model_sources": list(model.sources),
    }


def separate_audio(
    config: ProjectConfig,
    input_audio: Path,
    output_name: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_audio = Path(input_audio).expanduser().resolve()
    if not input_audio.is_file():
        raise FileNotFoundError(f"Audio input tidak ditemukan: {input_audio}")

    destination = estimate_track_directory(
        config.separated_root,
        config.model.name,
        output_name,
    )

    if _stems_complete(destination, config.stems) and not overwrite:
        return {
            "status": "skipped",
            "input": str(input_audio),
            "output": str(destination),
            "elapsed_seconds": 0.0,
        }

    started = time.perf_counter()
    details = _run_demucs_raw(config, input_audio, destination)
    elapsed = time.perf_counter() - started

    if not _stems_complete(destination, config.stems):
        raise RuntimeError(
            f"Demucs selesai tetapi output empat stem tidak lengkap di {destination}."
        )

    metadata = {
        "status": "completed",
        "input": str(input_audio),
        "output": str(destination),
        "model": config.model.name,
        "requested_device": config.model.device,
        "shifts": config.model.shifts,
        "overlap": config.model.overlap,
        "jobs": config.model.jobs,
        "segment": config.model.segment,
        "output_format": "WAV float32 raw model output; no rescale or clipping",
        "elapsed_seconds": elapsed,
        **details,
    }
    with (destination / "inference_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    return metadata
