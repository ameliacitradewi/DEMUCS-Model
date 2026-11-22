# Demucs V4 Music Source Separation

Pipeline Python untuk memisahkan audio musik dan mengevaluasi hasilnya pada MUSDB18-HQ. Proyek ini menjalankan model pralatih **Hybrid Transformer Demucs v4** (`htdemucs_ft`) pada satu campuran audio dan menghasilkan empat stem: **drums**, **bass**, **vocals**, dan **other**. Stem `other` mencakup semua sumber yang tidak masuk ke tiga kategori lainnya.

Pipeline mencakup validasi dataset, inferensi satu track atau satu split, evaluasi terhadap stem referensi, visualisasi sinyal, dan laporan agregat.

## Instalasi

Gunakan **Python 3.10 atau 3.11**. Dependensi utama mencakup Demucs 4.0.1, PyTorch, SoundFile, librosa, dan museval. Sediakan FFmpeg di `PATH` jika diperlukan untuk membaca format audio selain WAV. Bobot model diunduh pada inferensi pertama dan disimpan di cache.

### macOS dan Linux

```bash
git clone https://github.com/ameliacitradewi/DEMUCS-Model.git
cd DEMUCS-Model
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Windows

Clone repositori dan masuk ke direktorinya, lalu jalankan perintah berikut di Command Prompt. Skrip membuat virtual environment, memasang dependensi, dan menjalankan tes.

```bat
scripts\setup_windows.bat
.venv\Scripts\activate.bat
```

Tes juga dapat dijalankan dari virtual environment aktif dengan `python -m pytest -q`.

## Konfigurasi dataset dan model

Unduh [MUSDB18-HQ](https://zenodo.org/records/3338373), lalu atur lokasi dataset di [config.yaml](config.yaml). Path relatif dihitung dari direktori file konfigurasi.

```yaml
dataset_root: "../MUSDB18-HQ"
output_root: "outputs"
```

Dataset menggunakan struktur `<split>/<nama track>/`. Setiap folder track harus berisi `mixture.wav`, `drums.wav`, `bass.wav`, `vocals.wav`, dan `other.wav`. Distribusi resmi menyediakan split `train` dan `test`. Jika membuat split `valid` sendiri, ambil track dari split pelatihan dan pisahkan nama track antar-split.

Konfigurasi bawaan menggunakan `htdemucs_ft`, `shifts: 1`, dan `overlap: 0.25`. Mode `model.device: auto` memilih CUDA jika tersedia dan CPU jika tidak. Untuk mencoba MPS di Mac yang mendukungnya, atur `model.device: mps` secara eksplisit. Jika CUDA kehabisan memori, coba `model.segment: 7` atau gunakan CPU.

## Menjalankan pipeline

### Validasi dan evaluasi satu track

```bash
python run_project.py validate --splits train,test
python run_project.py list-tracks --split test
python run_project.py pipeline-one --split test
```

`validate` memeriksa keberadaan file, konsistensi sample rate, jumlah kanal dan frame, serta nama track yang muncul di lebih dari satu split. Tambahkan `--check-mixture-sum --sum-check-limit 5` untuk memeriksa apakah campuran mendekati penjumlahan stem referensi pada lima track pertama di setiap split.

`pipeline-one` menjalankan inferensi, evaluasi, dan visualisasi untuk track pertama menurut abjad. Gunakan `--track "Nama Folder Track"` untuk memilih track. Tambahkan `--bss` untuk BSS Eval v4 atau `--extras` untuk membuat plot FFT dan mel-spectrogram.

### Evaluasi seluruh split

```bash
python run_project.py separate-dataset --split test
python run_project.py evaluate-dataset --split test --bss
python run_project.py report --split test
```

Tambahkan `--limit 2` pada kedua perintah batch untuk mencoba alur pada dua track. Hapus `--bss` untuk menjalankan metrik default yang lebih cepat. Inferensi melewati track yang keempat WAV keluarannya sudah tersedia. Gunakan `--overwrite` pada perintah pemisahan untuk menghitung ulang setelah mengubah parameter model.

### Pisahkan lagu eksternal

```bash
python run_project.py external --audio "path/to/song.wav" --visualize
```

Perintah ini menghasilkan WAV dan, dengan `--visualize`, waveform serta spectrogram per stem. Untuk lagu tanpa stem referensi, gunakan listening test untuk menilai kebocoran antarsumber dan artefak. Metrik seperti SI-SDR dan BSS Eval memerlukan stem referensi.

## Metode evaluasi

Inferensi memakai API Python Demucs dan menyimpan estimasi sebagai **WAV float32**, tanpa rescaling atau hard clipping saat penulisan. Sebelum menghitung skor, evaluator menyesuaikan kanal dan sample rate estimasi dengan campuran lalu memotong seluruh sinyal ke panjang bersama terpendek. Perbandingan dilakukan antara estimasi dan referensi dengan label stem yang sama.

| Metrik | Interpretasi |
| --- | --- |
| SI-SDR dan SI-SDRi | SI-SDR mengukur kualitas estimasi tanpa penalti terhadap perubahan gain global. SI-SDRi mengukur peningkatan dibandingkan penggunaan campuran asli sebagai baseline. |
| Waveform SDR | Rasio energi referensi terhadap residual pada waveform yang sensitif terhadap perubahan gain. |
| BSS Eval v4 | Metrik opsional yang melaporkan median antar-frame untuk SDR, ISR, SIR, dan SAR. |
| Diagnostik | MAE, RMSE, korelasi, spectral convergence, log-spectral MAE, gain error, dan clipping. Konsistensi campuran mengukur selisih antara campuran asli dan jumlah estimasi stem. |

Visualisasi membandingkan waveform dan log-spectrogram dari referensi, estimasi, dan residual. Spectrogram menggunakan skala dB yang sama untuk ketiganya. Secara default, plot menampilkan segmen 10 detik mulai dari detik ke-30. Atur rentangnya dengan `--start` dan `--duration`.

## Hasil yang tercatat

Median per stem untuk `htdemucs_ft` pada split test MUSDB18-HQ, berdasarkan [catatan metodologi eksperimen](docs/methodology.md#results). Seluruh skor dalam dB.

| Stem | SI-SDR | SI-SDRi | BSS Eval SDR |
| --- | ---: | ---: | ---: |
| Bass | 8.54 | 14.55 | 10.30 |
| Drums | 9.80 | 13.95 | 9.86 |
| Other | 5.76 | 10.27 | 6.22 |
| Vocals | 9.24 | 14.97 | 9.47 |

Median SI-SDRi positif pada keempat stem. Drums memperoleh median SI-SDR tertinggi, sedangkan bass memperoleh median BSS Eval SDR tertinggi. Stem `other` memperoleh median terendah pada ketiga metrik.

## Keluaran eksperimen

| Lokasi di bawah `outputs/` | Isi |
| --- | --- |
| `reports/dataset_validation/` | `validation_tracks.csv` dan `validation_summary.json`. |
| `separated/<model>/<track>/` | Empat WAV estimasi dan `inference_metadata.json`. |
| `evaluation/<model>/<split>/` | CSV metrik dan ringkasan agregat, log kesalahan, serta `stem_metrics.csv` dan `track_summary.json` per track. |
| `figures/<model>/<split>/<track>/` | Plot waveform dan spectrogram untuk perintah visualisasi dataset. |
| `reports/<model>/<split>/` | Statistik per stem, peringkat track, contoh best/median/worst, boxplot, dan `automatic_findings.md`. |

Folder hasil pemisahan menggunakan nama model dan nama track tanpa menyertakan split. Untuk mencegah penggunaan ulang hasil dari audio berbeda dengan nama yang sama, gunakan `--name` yang unik untuk audio eksternal atau atur `output_root` terpisah untuk eksperimen lain.

## Struktur repositori

| Path | Peran |
| --- | --- |
| [run_project.py](run_project.py) | CLI untuk validasi, inferensi, evaluasi, visualisasi, dan laporan. |
| [config.yaml](config.yaml) | Pengaturan path, model, evaluasi, dan visualisasi. |
| [demucs_project/](demucs_project/) | Implementasi inferensi, pemrosesan audio, metrik, serta pembuatan plot dan laporan. |
| [docs/methodology.md](docs/methodology.md) | Desain eksperimen, definisi metrik, keterbatasan, dan hasil yang tercatat. |
| [scripts/](scripts/) dan [tests/](tests/) | Skrip instalasi dan pengujian. |
