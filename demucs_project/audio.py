from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


EPS = 1e-12


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Membaca audio sebagai float64 dengan bentuk (samples, channels)."""
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"Audio kosong: {path}")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"Audio mengandung NaN/Inf: {path}")
    return audio, int(sample_rate)


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio
    gcd = int(np.gcd(source_sr, target_sr))
    up = target_sr // gcd
    down = source_sr // gcd
    channels = [resample_poly(audio[:, idx], up=up, down=down) for idx in range(audio.shape[1])]
    return np.stack(channels, axis=1)


def match_channels(audio: np.ndarray, target_channels: int) -> np.ndarray:
    if audio.shape[1] == target_channels:
        return audio
    if audio.shape[1] == 1 and target_channels == 2:
        return np.repeat(audio, 2, axis=1)
    if target_channels == 1:
        return np.mean(audio, axis=1, keepdims=True)
    if audio.shape[1] > target_channels:
        return audio[:, :target_channels]
    raise ValueError(
        f"Tidak dapat mengubah {audio.shape[1]} channel menjadi {target_channels} channel."
    )


def align_length(*signals: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    """Memotong semua sinyal ke panjang minimum dan mengembalikan panjang awal."""
    original_lengths = [signal.shape[0] for signal in signals]
    minimum = min(original_lengths)
    return [signal[:minimum] for signal in signals], original_lengths


def to_mono(audio: np.ndarray) -> np.ndarray:
    return np.mean(audio, axis=1)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + EPS))


def peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio)))


def db(value: float, reference: float = 1.0) -> float:
    return float(20.0 * np.log10(max(value, EPS) / max(reference, EPS)))
