# Demucs V4 Music Source Separation

A Python pipeline for separating music into stems and evaluating the results on MUSDB18-HQ. It runs the pretrained **Hybrid Transformer Demucs v4** model (`htdemucs_ft`) on an audio mixture and produces four stems — **drums**, **bass**, **vocals**, and **other**. The `other` stem contains the remaining sources.

The pipeline covers dataset validation, single-track and batch inference, evaluation against reference stems, signal visualization, and aggregate reporting.

## Installation

Use **Python 3.10 or 3.11**. The main dependencies are Demucs 4.0.1, PyTorch, SoundFile, librosa, and museval. Make FFmpeg available on `PATH` if your input format requires it. The first inference run downloads and caches the pretrained model weights.

### macOS and Linux

```bash
git clone https://github.com/ameliacitradewi/DEMUCS-Model.git
cd DEMUCS-Model
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Windows

Clone the repository, open its directory in Command Prompt, and run the following commands. The setup script creates a virtual environment, installs dependencies, and runs the tests.

```bat
scripts\setup_windows.bat
.venv\Scripts\activate.bat
```

To run the tests separately, activate the virtual environment and run `python -m pytest -q`.

## Dataset and model configuration

Download [MUSDB18-HQ](https://zenodo.org/records/3338373) and set its location in [config.yaml](config.yaml). Relative paths are resolved from the directory containing the configuration file.

```yaml
dataset_root: "../MUSDB18-HQ"
output_root: "outputs"
```

Organize tracks as `<split>/<track name>/`. Each track directory must contain `mixture.wav`, `drums.wav`, `bass.wav`, `vocals.wav`, and `other.wav`. The official distribution includes `train` and `test` splits. If you create a `valid` split, draw its tracks from the original training split and keep track names unique across splits.

The default configuration uses `htdemucs_ft`, `shifts: 1`, and `overlap: 0.25`. With `model.device: auto`, the pipeline uses CUDA when available and falls back to CPU. To try MPS on a compatible Mac, set `model.device: mps` explicitly. If CUDA runs out of memory, try `model.segment: 7` or switch to CPU.

## Running the pipeline

### Validate the dataset and evaluate one track

```bash
python run_project.py validate --splits train,test
python run_project.py list-tracks --split test
python run_project.py pipeline-one --split test
```

`validate` checks that the required files exist, that sample rates, channel counts, and frame counts match within each track, and that track names are unique across the selected splits. Add `--check-mixture-sum --sum-check-limit 5` to check whether the mixture approximates the sum of the reference stems for the first five tracks in each split.

`pipeline-one` runs separation, evaluation, and visualization for the first track in alphabetical order. Use `--track "Track Folder Name"` to select a specific track. Add `--bss` to include BSS Eval v4 and `--extras` to generate FFT and mel spectrogram plots.

### Evaluate a full split

```bash
python run_project.py separate-dataset --split test
python run_project.py evaluate-dataset --split test --bss
python run_project.py report --split test
```

Add `--limit 2` to both batch commands for a two-track smoke test. Omit `--bss` to use the faster default metrics. Separation skips tracks whose four output WAV files already exist. Use `--overwrite` with the separation command to regenerate stems after changing model parameters.

### Separate an external audio file

```bash
python run_project.py external --audio "path/to/song.wav" --visualize
```

This command produces one WAV file per stem. With `--visualize`, it also generates a waveform and spectrogram for each stem. When reference stems are unavailable, use a listening test to assess source leakage and artifacts. SI-SDR and BSS Eval require reference stems.

## Evaluation methodology

The pipeline calls the Demucs Python API and writes the estimated stems as **32-bit floating-point WAV files**, preserving their amplitudes at export. Before scoring, the evaluator matches the estimates to the mixture’s sample rate and channel count, then trims all signals to the shortest length. Each estimate is compared with the reference stem that has the same label.

| Metric | Interpretation |
| --- | --- |
| SI-SDR and SI-SDRi | SI-SDR measures separation quality independently of overall gain. SI-SDRi measures the improvement over using the original mixture as the estimate. |
| Waveform SDR | Gain-sensitive ratio of reference energy to waveform residual energy. |
| BSS Eval v4 | Optional evaluation that reports frame-wise medians for SDR, ISR, SIR, and SAR. |
| Diagnostic metrics | MAE, RMSE, correlation, spectral convergence, log-spectral MAE, gain error, and clipping. Mixture consistency measures how closely the estimated stems sum to the original mixture. |

Plots compare the waveforms and log spectrograms of the reference, estimate, and residual. The spectrograms share a dB scale. By default, the plots show a 10-second segment starting 30 seconds into the track. Use `--start` and `--duration` to change the plotted segment.

## Benchmark results

The table shows median scores in dB for `htdemucs_ft` on the MUSDB18-HQ test split, as recorded in the [experiment methodology](docs/methodology.md#results).

| Stem | SI-SDR | SI-SDRi | BSS Eval SDR |
| --- | ---: | ---: | ---: |
| Bass | 8.54 | 14.55 | 10.30 |
| Drums | 9.80 | 13.95 | 9.86 |
| Other | 5.76 | 10.27 | 6.22 |
| Vocals | 9.24 | 14.97 | 9.47 |

Median SI-SDRi was positive for all four stems. Drums had the highest median SI-SDR, bass had the highest median BSS Eval SDR, and `other` had the lowest median on all three metrics.

## Experiment outputs

| Path under `outputs/` | Contents |
| --- | --- |
| `reports/dataset_validation/` | `validation_tracks.csv` and `validation_summary.json`. |
| `separated/<model>/<track>/` | Four estimated WAV stems and `inference_metadata.json`. |
| `evaluation/<model>/<split>/` | Aggregate metric and summary CSVs, an evaluation error log, and per-track `stem_metrics.csv` and `track_summary.json` files. |
| `figures/<model>/<split>/<track>/` | Waveform and spectrogram plots for dataset tracks. |
| `reports/<model>/<split>/` | Per-stem statistics, track rankings, best, median, and worst cases, box plots, and `automatic_findings.md`. |

Separation outputs are keyed by model and track name. Use a unique `--name` for external audio or a separate `output_root` for experiments with duplicate track names to avoid reusing an unrelated result.

## Repository structure

| Path | Purpose |
| --- | --- |
| [run_project.py](run_project.py) | CLI for validation, separation, evaluation, visualization, and reporting. |
| [config.yaml](config.yaml) | Paths and settings for the model, evaluation, and plots. |
| [demucs_project/](demucs_project/) | Inference, audio processing, metrics, plotting, and reporting. |
| [docs/methodology.md](docs/methodology.md) | Experiment design, metric definitions, limitations, and recorded results. |
| [scripts/](scripts/) and [tests/](tests/) | Setup scripts and tests. |
