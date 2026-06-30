import numpy as np

from demucs_project.metrics import correlation, si_sdr_db, spectral_metrics, waveform_sdr_db


def stereo(signal: np.ndarray) -> np.ndarray:
    return np.stack([signal, signal], axis=1)


def test_exact_signal_has_very_high_scores():
    rng = np.random.default_rng(7)
    reference = stereo(rng.standard_normal(44100))
    estimate = reference.copy()

    assert si_sdr_db(reference, estimate) > 100
    assert waveform_sdr_db(reference, estimate) > 100
    assert correlation(reference, estimate) > 0.999999


def test_si_sdr_is_scale_invariant_but_waveform_sdr_is_not():
    rng = np.random.default_rng(11)
    reference = stereo(rng.standard_normal(22050))
    estimate = 0.5 * reference

    assert si_sdr_db(reference, estimate) > 100
    assert waveform_sdr_db(reference, estimate) < 10


def test_spectral_metrics_are_near_zero_for_exact_signal():
    sample_rate = 8000
    time = np.arange(sample_rate) / sample_rate
    reference = stereo(np.sin(2 * np.pi * 440 * time))
    convergence, log_mae = spectral_metrics(reference, reference, n_fft=512, hop_length=128)

    assert convergence < 1e-10
    assert log_mae < 1e-10
