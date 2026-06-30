# Proyek Demucs V4 untuk Pemisahan Instrumen Musik

Proyek ini menjalankan **Hybrid Transformer Demucs V4 (`htdemucs_ft`)** pada `mixture.wav`, membandingkan empat stem hasil estimasi dengan ground truth MUSDB18-HQ, membuat visualisasi waveform/spectrogram, menghitung metrik, dan mengagregasi hasil seluruh test set.

Empat stem yang dievaluasi:

- `drums.wav`
- `bass.wav`
- `other.wav`
- `vocals.wav`

> `other` adalah gabungan semua sumber selain drums, bass, dan vocals. Ia bukan satu instrumen tunggal.

## 1. Struktur proyek

```text
.
├── config.yaml
├── requirements.txt
├── run_project.py
├── demucs_project/
│   ├── audio.py
│   ├── config.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── metrics.py
│   ├── paths.py
│   ├── reporting.py
│   ├── validation.py
│   └── visualization.py
├── docs/
│   └── methodology.md
├── scripts/
├── tests/
└── .vscode/
```

Output eksperimen dibuat otomatis di folder `outputs/`.

## 2. Prasyarat

Gunakan **Python 3.10 atau 3.11**. Python 3.13 tidak direkomendasikan untuk stack Demucs 4.0.1 lama.

Untuk input selain WAV, FFmpeg sebaiknya tersedia pada `PATH`. MUSDB18-HQ berbentuk WAV sehingga `soundfile` dapat membaca ground truth secara langsung, tetapi Demucs dapat tetap menggunakan FFmpeg/torchaudio saat inferensi.

### Windows

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pytest -q
```

Atau jalankan:

```powershell
scripts\setup_windows.bat
```

### Linux/macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pytest -q
```

### GPU NVIDIA

`requirements.txt` membiarkan `demucs` memasang dependency PyTorch. Untuk kontrol penuh atas CUDA, instal `torch` dan `torchaudio` lebih dahulu memakai perintah dari selector resmi PyTorch, kemudian jalankan `pip install -r requirements.txt`.

Periksa GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 3. Konfigurasi dataset

Edit `config.yaml`:

```yaml
dataset_root: "D:/datasets/MUSDB18-HQ"
output_root: "outputs"
```

Struktur yang diharapkan:

```text
MUSDB18-HQ/
├── train/
│   └── Nama Lagu/
│       ├── mixture.wav
│       ├── drums.wav
│       ├── bass.wav
│       ├── other.wav
│       └── vocals.wav
├── valid/
│   └── ...
└── test/
    └── ...
```

MUSDB18-HQ resmi memiliki 100 track train dan 50 track test. Struktur 50 train + 50 valid + 50 test dapat digunakan apabila folder valid dibuat dari 100 track train resmi dan tidak ada overlap nama/track dengan test.

## 4. Urutan eksperimen

### A. Validasi dataset

```bash
python run_project.py validate
```

Pemeriksaan yang dilakukan:

- keberadaan lima WAV;
- sample rate antar-file;
- jumlah channel;
- jumlah frame;
- nama track yang tumpang tindih antar-split.

Pemeriksaan tambahan bahwa mixture mendekati penjumlahan empat stem:

```bash
python run_project.py validate --check-mixture-sum --sum-check-limit 5
```

Jangan langsung memeriksa 150 lagu bila hanya ingin smoke test, karena seluruh audio harus dibaca.

### B. Lihat nama track test

```bash
python run_project.py list-tracks --split test
```

### C. Jalankan satu track lengkap

Tanpa `--track`, program memakai track pertama secara alfabetis:

```bash
python run_project.py pipeline-one --split test
```

Dengan nama track tertentu:

```bash
python run_project.py pipeline-one --split test --track "Nama Folder Lagu"
```

Perintah tersebut melakukan:

1. inferensi `mixture.wav` dengan `htdemucs_ft`;
2. evaluasi reference vs estimate;
3. pembuatan waveform dan log-spectrogram untuk setiap stem.

Tambahkan BSS Eval v4:

```bash
python run_project.py pipeline-one --split test --track "Nama Folder Lagu" --bss
```

BSS Eval lebih lambat daripada metrik default. Gunakan pada satu track dahulu untuk memastikan environment stabil.

Tambahkan FFT dan mel-spectrogram diagnostik:

```bash
python run_project.py pipeline-one --split test --track "Nama Folder Lagu" --extras
```

### D. Inferensi seluruh test set

```bash
python run_project.py separate-dataset --split test
```

Smoke test dua track:

```bash
python run_project.py separate-dataset --split test --limit 2
```

Output disimpan sebagai:

```text
outputs/separated/htdemucs_ft/<nama track>/
├── drums.wav
├── bass.wav
├── other.wav
├── vocals.wav
└── inference_metadata.json
```

Semua input dataset bernama `mixture.wav`. Kode menggunakan nama folder lagu untuk output sehingga hasil antar-track tidak tertimpa.

### E. Evaluasi seluruh test set

Metrik cepat dan diagnostik:

```bash
python run_project.py evaluate-dataset --split test
```

Dengan BSS Eval SDR/ISR/SIR/SAR:

```bash
python run_project.py evaluate-dataset --split test --bss
```

Output utama:

```text
outputs/evaluation/htdemucs_ft/test/
├── all_stem_metrics.csv
├── all_track_summaries.csv
├── evaluation_errors.csv
└── <nama track>/
    ├── stem_metrics.csv
    └── track_summary.json
```

### F. Buat report agregat

```bash
python run_project.py report --split test
```

Report menyertakan:

- median, mean, dan standard deviation per stem;
- skor per track;
- best, median, dan worst case;
- boxplot distribusi metrik;
- `automatic_findings.md` sebagai draft interpretasi.

## 5. Metrik yang dihitung

### Default

| Metrik | Makna | Arah yang baik |
|---|---|---|
| SI-SDR | kemiripan estimate terhadap target tanpa penalti gain global | tinggi |
| SI-SDRi | peningkatan SI-SDR terhadap mixture | positif/tinggi |
| Waveform SDR | rasio energi reference terhadap error secara scale-dependent | tinggi |
| MAE/RMSE | error waveform | rendah |
| Correlation | kemiripan bentuk sinyal | tinggi |
| Spectral convergence | error magnitude STFT relatif | rendah |
| Log-spectral MAE | perbedaan magnitude STFT dalam dB | rendah |
| Gain error | selisih RMS estimate vs reference | mendekati 0 dB |
| Clip fraction | proporsi sample estimate di luar/tepat batas | rendah |
| Mixture consistency | selisih mixture dengan jumlah empat estimate | rendah |

`waveform_sdr_db` pada proyek ini adalah perbandingan scale-dependent langsung dan **bukan** BSS Eval SDR.

### Opsional dengan `--bss`

| Metrik | Makna |
|---|---|
| BSS SDR | distorsi keseluruhan |
| ISR | distorsi spatial/stereo image |
| SIR | interference atau kebocoran sumber lain |
| SAR | artefak yang dibuat model |

Semua skor BSS disimpan sebagai median antar-frame.

## 6. Visualisasi

Default menghasilkan dua gambar per stem:

1. **Waveform ground truth, estimate, dan residual**
2. **Log-magnitude spectrogram ground truth, estimate, dan residual**

Skala dB spectrogram dibuat bersama agar reference dan estimate dapat dibandingkan secara adil.

Waveform berguna untuk clipping, transient, gain, dan kesalahan waktu. Spectrogram lebih berguna untuk melihat leakage frekuensi, frekuensi yang hilang, dan artefak.

FFT dan mel-spectrogram tersedia sebagai diagnostik tambahan melalui `--extras`. MFCC dan chromagram tidak dijadikan evaluasi utama karena keduanya membuang atau merangkum detail yang penting bagi source separation; khususnya chromagram juga kurang relevan untuk drum.

## 7. Lagu pop eksternal

Pisahkan lagu eksternal:

```bash
python run_project.py external --audio "D:/musik/lagu_pop.wav"
```

Tanpa ground-truth stems, Anda **tidak dapat** menghitung SDR, SI-SDR, SIR, SAR, atau ISR secara valid. Gunakan listening test terstruktur, misalnya skor 1–5 untuk:

- target preservation;
- leakage/interference;
- artifacts;
- naturalness;
- usability.

Untuk generalisasi yang lebih kuat, gunakan lagu eksternal yang memiliki stems asli dan evaluasi dengan pipeline yang sama.

## 8. Kriteria kesimpulan

Model dapat disebut layak sebagai prototype apabila:

- proses inferensi stabil pada seluruh test set;
- mayoritas nilai SI-SDRi positif;
- performa tidak gagal sistematis pada salah satu stem;
- best/median/worst case telah dianalisis;
- artefak pada spectrogram dan listening test masih dapat diterima;
- waktu dan resource inferensi sesuai kebutuhan.

Jangan menyimpulkan “layak untuk seluruh lagu pop” hanya dari satu lagu atau hanya dari MUSDB18-HQ. Tambahkan external test dan nyatakan keterbatasan model.

## 9. Catatan model

- `htdemucs_ft` adalah versi fine-tuned dari `htdemucs` dan lebih lambat karena merupakan bag beberapa model.
- `shifts` meningkatkan stabilitas tetapi menambah waktu komputasi. Nilai default proyek adalah 1.
- `overlap=0.25` mengikuti default Demucs.
- Inferensi memakai core Python Demucs dan menyimpan WAV float32 mentah melalui `soundfile`, sehingga tidak ada rescale atau hard clipping tersembunyi sebelum evaluasi. Pendekatan ini juga menghindari keterbatasan CLI pada rilis PyPI 4.0.1.
- Hindari `mdx_extra` untuk evaluasi murni MUSDB test karena model tersebut diketahui menggunakan MUSDB test sebagai extra training data.
- Model `htdemucs`/`htdemucs_ft` menggunakan MUSDB dan data tambahan, sehingga status benar-benar unseen untuk lagu eksternal tidak selalu dapat diverifikasi.

## 10. Troubleshooting

### CUDA out of memory

Edit `config.yaml`:

```yaml
model:
  device: "cuda"
  segment: 7
  shifts: 1
```

Atau gunakan CPU:

```yaml
model:
  device: "cpu"
```

### Model diunduh saat run pertama

Normal. Bobot pretrained Demucs akan diunduh dan disimpan dalam cache lokal.

### `museval` sangat lambat

Jalankan evaluasi default tanpa `--bss` untuk seluruh dataset. Gunakan `--bss` pada subset atau saat hasil final diperlukan.

### Output tidak ditemukan

Pastikan inferensi sudah selesai dan struktur berikut ada:

```text
outputs/separated/htdemucs_ft/<nama track>/<stem>.wav
```

### Ulangi inferensi

Gunakan:

```bash
python run_project.py separate-one --split test --track "Nama Folder Lagu" --overwrite
```
