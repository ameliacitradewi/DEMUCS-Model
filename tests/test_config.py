from pathlib import Path

from demucs_project.config import load_config


def test_load_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset_root: dataset
output_root: output
model:
  name: htdemucs_ft
  device: cpu
  shifts: 1
  overlap: 0.25
  jobs: 0
  segment: null
stems: [drums, bass, other, vocals]
evaluation:
  n_fft: 1024
  hop_length: 256
  top_db: 80
  bss_window_seconds: 2.0
  bss_hop_seconds: 1.5
  bss_filter_length: 512
visualization:
  start_seconds: 0
  duration_seconds: 5
  n_fft: 1024
  hop_length: 256
  top_db: 80
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.model.name == "htdemucs_ft"
    assert config.dataset_root == (tmp_path / "dataset").resolve()
    assert config.output_root == (tmp_path / "output").resolve()
    assert config.stems == ("drums", "bass", "other", "vocals")
