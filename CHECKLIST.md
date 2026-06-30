# Checklist Eksekusi dan Laporan

## Setup

- [ ] Environment Python 3.10/3.11 aktif
- [ ] `pip install -r requirements.txt` berhasil
- [ ] `python -m pytest -q` lulus
- [ ] `dataset_root` pada `config.yaml` sudah benar
- [ ] GPU terdeteksi bila akan memakai CUDA

## Validasi data

- [ ] Lima WAV tersedia untuk setiap track
- [ ] Audio stereo 44,1 kHz
- [ ] Tidak ada overlap track train/valid/test
- [ ] Mixture sum check dilakukan pada sampel track

## Demonstrasi satu track

- [ ] Inferensi berhasil
- [ ] Empat estimate WAV tersimpan
- [ ] SI-SDR, SI-SDRi, dan metrik spektral dihitung
- [ ] BSS SDR/ISR/SIR/SAR dicoba
- [ ] Waveform dan spectrogram dianalisis
- [ ] Leakage serta artifacts dicatat

## Evaluasi test set

- [ ] Seluruh 50 test track diproses
- [ ] Error/failure log diperiksa
- [ ] Median dan mean per stem dilaporkan
- [ ] Proporsi SI-SDRi positif dilaporkan
- [ ] Best, median, dan worst case dianalisis
- [ ] Waktu inferensi dan resource dicatat bila relevan

## External pop

- [ ] Beberapa lagu eksternal dipisahkan
- [ ] Status ground truth dijelaskan
- [ ] Listening test memakai kriteria konsisten
- [ ] Domain shift dan keterbatasan dibahas

## Kesimpulan

- [ ] Klaim dibatasi pada bukti yang tersedia
- [ ] `other` dijelaskan sebagai kategori gabungan
- [ ] Data tambahan training Demucs disebutkan sebagai keterbatasan unseen test
- [ ] Tidak menyimpulkan kualitas universal hanya dari satu lagu
