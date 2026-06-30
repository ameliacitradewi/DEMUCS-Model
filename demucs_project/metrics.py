from __future__ import annotations

from typing import Any

import librosa
import numpy as np

from .audio import EPS, db, peak, rms


def _mean_channel_score(reference: np.ndarray, estimate: np.ndarray, scorer) -> float:
    scores: list[float] = []
    for channel in range(reference.shape[1]):
        scores.append(float(scorer(reference[:, channel], estimate[:, channel])))
    return float(np.mean(scores))


def si_sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-Invariant SDR, dirata-ratakan antar-channel."""

    def scorer(ref: np.ndarray, est: np.ndarray) -> float:
        ref = ref - np.mean(ref)
        est = est - np.mean(est)
        scale = np.dot(est, ref) / (np.dot(ref, ref) + EPS)
        target = scale * ref
        error = est - target
        return 10.0 * np.log10(
            (np.sum(target**2) + EPS) / (np.sum(error**2) + EPS)
        )

    return _mean_channel_score(reference, estimate, scorer)


def waveform_sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """SDR scale-dependent sederhana; bukan BSS Eval SDR."""

    def scorer(ref: np.ndarray, est: np.ndarray) -> float:
        return 10.0 * np.log10(
            (np.sum(ref**2) + EPS) / (np.sum((ref - est) ** 2) + EPS)
        )

    return _mean_channel_score(reference, estimate, scorer)


def correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    def scorer(ref: np.ndarray, est: np.ndarray) -> float:
        ref_std = float(np.std(ref))
        est_std = float(np.std(est))
        if ref_std < EPS or est_std < EPS:
            return float("nan")
        return float(np.corrcoef(ref, est)[0, 1])

    values = []
    for channel in range(reference.shape[1]):
        value = scorer(reference[:, channel], estimate[:, channel])
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def spectral_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    n_fft: int,
    hop_length: int,
    top_db: float = 80.0,
    chunk_samples: int = 1_000_000,
) -> tuple[float, float]:
    """Metrik STFT pada downmix mono, diproses per chunk agar hemat memori."""
    reference_mono = np.mean(reference, axis=1).astype(np.float32, copy=False)
    estimate_mono = np.mean(estimate, axis=1).astype(np.float32, copy=False)

    error_energy = 0.0
    reference_energy = 0.0
    log_error_sum = 0.0
    log_value_count = 0

    for start in range(0, len(reference_mono), chunk_samples):
        ref_chunk = reference_mono[start : start + chunk_samples]
        est_chunk = estimate_mono[start : start + chunk_samples]
        if len(ref_chunk) < n_fft:
            pad = n_fft - len(ref_chunk)
            ref_chunk = np.pad(ref_chunk, (0, pad))
            est_chunk = np.pad(est_chunk, (0, pad))

        ref_stft = np.abs(
            librosa.stft(
                ref_chunk,
                n_fft=n_fft,
                hop_length=hop_length,
                center=False,
            )
        )
        est_stft = np.abs(
            librosa.stft(
                est_chunk,
                n_fft=n_fft,
                hop_length=hop_length,
                center=False,
            )
        )

        difference = est_stft - ref_stft
        error_energy += float(np.sum(np.square(difference), dtype=np.float64))
        reference_energy += float(np.sum(np.square(ref_stft), dtype=np.float64))

        shared_reference = max(float(np.max(ref_stft)), EPS)
        floor_db = -float(top_db)
        ref_log = np.maximum(
            20.0 * np.log10(np.maximum(ref_stft, EPS) / shared_reference),
            floor_db,
        )
        est_log = np.maximum(
            20.0 * np.log10(np.maximum(est_stft, EPS) / shared_reference),
            floor_db,
        )
        log_error_sum += float(
            np.sum(np.abs(est_log - ref_log), dtype=np.float64)
        )
        log_value_count += int(ref_log.size)

    spectral_convergence = np.sqrt(error_energy) / (np.sqrt(reference_energy) + EPS)
    log_spectral_mae = log_error_sum / max(log_value_count, 1)
    return float(spectral_convergence), float(log_spectral_mae)


def evaluate_stem_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    mixture: np.ndarray,
    n_fft: int,
    hop_length: int,
    top_db: float = 80.0,
) -> dict[str, Any]:
    current_si_sdr = si_sdr_db(reference, estimate)
    mixture_si_sdr = si_sdr_db(reference, mixture)
    spectral_convergence, log_spectral_mae_db = spectral_metrics(
        reference,
        estimate,
        n_fft=n_fft,
        hop_length=hop_length,
        top_db=top_db,
    )

    reference_rms = rms(reference)
    estimate_rms = rms(estimate)

    return {
        "si_sdr_db": current_si_sdr,
        "mixture_si_sdr_db": mixture_si_sdr,
        "si_sdri_db": current_si_sdr - mixture_si_sdr,
        "waveform_sdr_db": waveform_sdr_db(reference, estimate),
        "mae": float(np.mean(np.abs(estimate - reference))),
        "rmse": float(np.sqrt(np.mean((estimate - reference) ** 2))),
        "correlation": correlation(reference, estimate),
        "spectral_convergence": spectral_convergence,
        "log_spectral_mae_db": log_spectral_mae_db,
        "reference_rms_dbfs": db(reference_rms),
        "estimate_rms_dbfs": db(estimate_rms),
        "gain_error_db": db(estimate_rms, reference_rms),
        "reference_peak": peak(reference),
        "estimate_peak": peak(estimate),
        "estimate_clip_fraction": float(np.mean(np.abs(estimate) >= 1.0)),
    }


def finite_median(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def bss_eval_v4(
    references: np.ndarray,
    estimates: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
    filter_length: int,
) -> dict[str, np.ndarray]:
    """Menghitung BSS Eval v4 melalui museval secara opsional."""
    try:
        import museval
    except ImportError as exc:
        raise RuntimeError(
            "museval belum terpasang. Jalankan: pip install museval"
        ) from exc

    window = max(1, int(round(window_seconds * sample_rate)))
    hop = max(1, int(round(hop_seconds * sample_rate)))

    sdr, isr, sir, sar, _ = museval.metrics.bss_eval(
        references,
        estimates,
        window=window,
        hop=hop,
        compute_permutation=False,
        filters_len=filter_length,
        framewise_filters=False,
        bsseval_sources_version=False,
    )
    return {"sdr": sdr, "isr": isr, "sir": sir, "sar": sar}
