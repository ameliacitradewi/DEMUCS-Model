from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .audio import EPS, load_audio, match_channels, resample_audio, to_mono
from .config import ProjectConfig
from .paths import estimate_track_directory


def _load_aligned_segment(
    reference_path: Path,
    estimate_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    reference, sample_rate = load_audio(reference_path)
    estimate, estimate_sr = load_audio(estimate_path)
    estimate = match_channels(estimate, reference.shape[1])
    estimate = resample_audio(estimate, estimate_sr, sample_rate)

    minimum = min(len(reference), len(estimate))
    reference = reference[:minimum]
    estimate = estimate[:minimum]

    start_sample = max(0, int(round(start_seconds * sample_rate)))
    if start_sample >= minimum:
        start_sample = 0
    end_sample = min(minimum, start_sample + int(round(duration_seconds * sample_rate)))
    if end_sample <= start_sample:
        raise ValueError("Segmen visualisasi kosong.")

    actual_start = start_sample / sample_rate
    return (
        to_mono(reference[start_sample:end_sample]),
        to_mono(estimate[start_sample:end_sample]),
        sample_rate,
        actual_start,
    )


def plot_waveform_comparison(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    actual_start: float,
    title: str,
    output_path: Path,
) -> None:
    residual = estimate - reference
    times = actual_start + np.arange(len(reference)) / sample_rate
    amplitude_limit = max(
        float(np.max(np.abs(reference))),
        float(np.max(np.abs(estimate))),
        float(np.max(np.abs(residual))),
        1e-3,
    )

    figure, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(times, reference, linewidth=0.7)
    axes[0].set_title("Ground truth")
    axes[1].plot(times, estimate, linewidth=0.7)
    axes[1].set_title("Estimasi Demucs")
    axes[2].plot(times, residual, linewidth=0.7)
    axes[2].set_title("Residual = estimasi - ground truth")

    for axis in axes:
        axis.set_ylim(-amplitude_limit, amplitude_limit)
        axis.set_ylabel("Amplitudo")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Waktu (detik)")
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _magnitude_stft(signal: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    return np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))


def _shared_db(magnitudes: list[np.ndarray], top_db: float) -> list[np.ndarray]:
    shared_reference = max(float(np.max(magnitude)) for magnitude in magnitudes)
    shared_reference = max(shared_reference, EPS)
    outputs = []
    for magnitude in magnitudes:
        values = 20.0 * np.log10(np.maximum(magnitude, EPS) / shared_reference)
        outputs.append(np.maximum(values, -top_db))
    return outputs


def plot_spectrogram_comparison(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    actual_start: float,
    n_fft: int,
    hop_length: int,
    top_db: float,
    title: str,
    output_path: Path,
) -> None:
    residual = estimate - reference
    magnitudes = [
        _magnitude_stft(reference, n_fft, hop_length),
        _magnitude_stft(estimate, n_fft, hop_length),
        _magnitude_stft(residual, n_fft, hop_length),
    ]
    db_values = _shared_db(magnitudes, top_db=top_db)

    figure, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, sharey=True)
    labels = ["Ground truth", "Estimasi Demucs", "Residual"]
    image = None
    for axis, values, label in zip(axes, db_values, labels):
        image = librosa.display.specshow(
            values,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="log",
            vmin=-top_db,
            vmax=0,
            ax=axis,
        )
        axis.set_title(label)
        axis.set_xlabel("")
    axes[-1].set_xlabel(f"Waktu relatif dari {actual_start:.2f} detik")
    if image is not None:
        figure.colorbar(image, ax=axes, format="%+2.0f dB", pad=0.01)
    figure.suptitle(title)
    figure.subplots_adjust(top=0.93, hspace=0.25, right=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_global_spectrum(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    title: str,
    output_path: Path,
) -> None:
    window = np.hanning(len(reference))
    frequencies = np.fft.rfftfreq(len(reference), d=1.0 / sample_rate)
    ref_spectrum = np.abs(np.fft.rfft(reference * window))
    est_spectrum = np.abs(np.fft.rfft(estimate * window))
    ref_db = 20.0 * np.log10(np.maximum(ref_spectrum, EPS))
    est_db = 20.0 * np.log10(np.maximum(est_spectrum, EPS))

    figure, axis = plt.subplots(figsize=(14, 5))
    axis.semilogx(frequencies[1:], ref_db[1:], label="Ground truth", linewidth=1.0)
    axis.semilogx(frequencies[1:], est_db[1:], label="Estimasi", linewidth=1.0)
    axis.set_xlabel("Frekuensi (Hz)")
    axis.set_ylabel("Magnitude (dB)")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_mel_comparison(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    top_db: float,
    title: str,
    output_path: Path,
) -> None:
    residual = estimate - reference
    signals = [reference, estimate, residual]
    mel_values = [
        librosa.feature.melspectrogram(
            y=signal,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=128,
            power=2.0,
        )
        for signal in signals
    ]
    shared_reference = max(float(np.max(value)) for value in mel_values)
    db_values = [
        librosa.power_to_db(value, ref=max(shared_reference, EPS), top_db=top_db)
        for value in mel_values
    ]

    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    labels = ["Ground truth", "Estimasi Demucs", "Residual"]
    image = None
    for axis, values, label in zip(axes, db_values, labels):
        image = librosa.display.specshow(
            values,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            vmin=-top_db,
            vmax=0,
            ax=axis,
        )
        axis.set_title(label)
    if image is not None:
        figure.colorbar(image, ax=axes, format="%+2.0f dB", pad=0.01)
    figure.suptitle(title)
    figure.subplots_adjust(top=0.93, hspace=0.3, right=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def visualize_track(
    config: ProjectConfig,
    split: str,
    track_name: str,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    include_extras: bool = False,
) -> list[Path]:
    start = (
        config.visualization.start_seconds
        if start_seconds is None
        else start_seconds
    )
    duration = (
        config.visualization.duration_seconds
        if duration_seconds is None
        else duration_seconds
    )

    reference_dir = config.dataset_root / split / track_name
    estimate_dir = estimate_track_directory(
        config.separated_root,
        config.model.name,
        track_name,
    )
    output_dir = config.figures_root / config.model.name / split / track_name
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for stem in config.stems:
        reference, estimate, sample_rate, actual_start = _load_aligned_segment(
            reference_dir / f"{stem}.wav",
            estimate_dir / f"{stem}.wav",
            start_seconds=start,
            duration_seconds=duration,
        )
        title = f"{track_name} — {stem} — segmen {actual_start:.2f}s"

        waveform_path = output_dir / f"{stem}_waveform.png"
        plot_waveform_comparison(
            reference,
            estimate,
            sample_rate,
            actual_start,
            title,
            waveform_path,
        )
        created.append(waveform_path)

        spectrogram_path = output_dir / f"{stem}_spectrogram.png"
        plot_spectrogram_comparison(
            reference,
            estimate,
            sample_rate,
            actual_start,
            n_fft=config.visualization.n_fft,
            hop_length=config.visualization.hop_length,
            top_db=config.visualization.top_db,
            title=title,
            output_path=spectrogram_path,
        )
        created.append(spectrogram_path)

        if include_extras:
            spectrum_path = output_dir / f"{stem}_fft_spectrum.png"
            plot_global_spectrum(
                reference,
                estimate,
                sample_rate,
                title=f"FFT global segmen — {title}",
                output_path=spectrum_path,
            )
            created.append(spectrum_path)

            mel_path = output_dir / f"{stem}_mel_spectrogram.png"
            plot_mel_comparison(
                reference,
                estimate,
                sample_rate,
                n_fft=config.visualization.n_fft,
                hop_length=config.visualization.hop_length,
                top_db=config.visualization.top_db,
                title=f"Mel-spectrogram diagnostik — {title}",
                output_path=mel_path,
            )
            created.append(mel_path)

    return created


def plot_waveform_single(
    signal: np.ndarray,
    sample_rate: int,
    actual_start: float,
    title: str,
    output_path: Path,
) -> None:
    times = actual_start + np.arange(len(signal)) / sample_rate
    amplitude_limit = max(float(np.max(np.abs(signal))), 1e-3)

    figure, axis = plt.subplots(figsize=(14, 4))
    axis.plot(times, signal, linewidth=0.7)
    axis.set_ylim(-amplitude_limit, amplitude_limit)
    axis.set_ylabel("Amplitudo")
    axis.set_xlabel("Waktu (detik)")
    axis.grid(alpha=0.2)
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_spectrogram_single(
    signal: np.ndarray,
    sample_rate: int,
    actual_start: float,
    n_fft: int,
    hop_length: int,
    top_db: float,
    title: str,
    output_path: Path,
) -> None:
    magnitude = _magnitude_stft(signal, n_fft, hop_length)
    ref_value = max(float(np.max(magnitude)), EPS)
    db_values = 20.0 * np.log10(np.maximum(magnitude, EPS) / ref_value)
    db_values = np.maximum(db_values, -top_db)

    figure, axis = plt.subplots(figsize=(14, 5))
    image = librosa.display.specshow(
        db_values,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
        vmin=-top_db,
        vmax=0,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel(f"Waktu relatif dari {actual_start:.2f} detik")
    figure.colorbar(image, ax=axis, format="%+2.0f dB", pad=0.01)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def visualize_external_track(
    config: ProjectConfig,
    separated_dir: Path,
    output_name: str,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> list[Path]:
    start = (
        config.visualization.start_seconds
        if start_seconds is None
        else start_seconds
    )
    duration = (
        config.visualization.duration_seconds
        if duration_seconds is None
        else duration_seconds
    )

    output_dir = separated_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for stem in config.stems:
        stem_path = separated_dir / f"{stem}.wav"
        if not stem_path.exists():
            continue

        signal, sample_rate = load_audio(stem_path)
        signal = to_mono(signal)

        minimum = len(signal)
        start_sample = max(0, int(round(start * sample_rate)))
        if start_sample >= minimum:
            start_sample = 0
        end_sample = min(minimum, start_sample + int(round(duration * sample_rate)))
        if end_sample <= start_sample:
            continue

        actual_start = start_sample / sample_rate
        segment = signal[start_sample:end_sample]

        title = f"{output_name} — {stem} — segmen {actual_start:.2f}s"

        waveform_path = output_dir / f"{stem}_waveform.png"
        plot_waveform_single(
            segment,
            sample_rate,
            actual_start,
            title,
            waveform_path,
        )
        created.append(waveform_path)

        spectrogram_path = output_dir / f"{stem}_spectrogram.png"
        plot_spectrogram_single(
            segment,
            sample_rate,
            actual_start,
            n_fft=config.visualization.n_fft,
            hop_length=config.visualization.hop_length,
            top_db=config.visualization.top_db,
            title=title,
            output_path=spectrogram_path,
        )
        created.append(spectrogram_path)

    return created

