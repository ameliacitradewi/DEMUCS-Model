# Metodologi Eksperimen

## Pertanyaan penelitian

1. Seberapa dekat empat stem hasil `htdemucs_ft` terhadap ground truth MUSDB18-HQ?
2. Stem mana yang paling mudah dan paling sulit dipisahkan?
3. Apakah hasil model secara konsisten lebih baik daripada memakai mixture sebagai estimasi target?
4. Apakah error lebih dominan berupa leakage sumber lain, artefak, kesalahan gain, atau kehilangan detail spektral?
5. Seberapa jauh hasil benchmark dapat mendukung penggunaan pada lagu pop eksternal?

## Desain eksperimen

### Unit analisis

Satu unit analisis adalah pasangan `(track, stem)`. Empat stem pada setiap track dievaluasi secara terpisah.

### Input dan target

- Input model: `mixture.wav`
- Ground truth: `drums.wav`, `bass.wav`, `other.wav`, dan `vocals.wav`
- Estimate: empat WAV keluaran Demucs

### Kontrol eksperimen

- model dan parameter inferensi yang sama untuk semua track;
- output float32 tanpa rescaling atau hard clipping;
- sample rate estimate disesuaikan ke reference bila diperlukan;
- panjang sinyal disejajarkan dengan pemotongan ke panjang minimum;
- source permutation dimatikan karena label output Demucs sudah diketahui;
- test set tidak dipakai untuk memilih konfigurasi bila konfigurasi dibandingkan.

## Metrik

### SI-SDR dan SI-SDR improvement

SI-SDR memproyeksikan estimate pada reference, sehingga perbedaan gain global tidak dihukum. SI-SDRi membandingkan hasil pemisahan terhadap baseline mixture:

```text
SI-SDRi = SI-SDR(estimate, reference) - SI-SDR(mixture, reference)
```

SI-SDRi positif berarti estimate lebih dekat pada sumber target daripada mixture asli.

### Scale-dependent waveform SDR

Proyek juga menghitung rasio energi reference terhadap residual langsung. Metrik ini sensitif terhadap gain dan tidak boleh disamakan dengan BSS Eval SDR.

### BSS Eval v4

Saat `--bss` aktif, proyek menghitung median frame-wise:

- SDR: distorsi total;
- ISR: distorsi spatial image;
- SIR: interference sumber lain;
- SAR: artefak.

### Metrik diagnostik

MAE, RMSE, correlation, spectral convergence, log-spectral MAE, gain error, peak, clip fraction, dan mixture consistency membantu menjelaskan penyebab skor utama.

## Visualisasi

Visualisasi utama menggunakan segmen 10 detik agar detail tetap terbaca:

- waveform reference;
- waveform estimate;
- waveform residual;
- log-spectrogram reference;
- log-spectrogram estimate;
- log-spectrogram residual.

Gunakan segmen yang sama dan skala dB yang sama untuk perbandingan adil.

## Analisis dataset penuh

Laporkan minimal:

- median dan mean per stem;
- standard deviation atau IQR;
- proporsi SI-SDRi positif;
- jumlah track gagal;
- best, median, dan worst case;
- waktu inferensi bila dibutuhkan;
- contoh leakage dan artifacts dari listening/spectrogram.

Median berguna karena distribusi skor source separation dapat dipengaruhi outlier dan bagian silent.

## Validitas dan keterbatasan

1. Satu track hanya cukup sebagai demonstrasi pipeline.
2. Test MUSDB18-HQ menilai benchmark distribution, bukan seluruh lagu pop dunia.
3. Lagu komersial tanpa stems hanya dapat dinilai secara subjektif atau dengan metrik non-reference yang tidak membuktikan kebenaran pemisahan.
4. `other` mencampur berbagai instrumen dan biasanya lebih ambigu.
5. Data tambahan training Demucs membatasi kemampuan memastikan lagu eksternal benar-benar unseen.
6. Listening test sebaiknya memakai beberapa pendengar dan urutan sampel yang disamarkan.

## Results
Median per stem:
        si_sdr_db  si_sdri_db  bss_sdr_db
stem                                     
bass       8.5436     14.5487     10.2955
drums      9.8048     13.9490      9.8622
other      5.7606     10.2731      6.2208
vocals     9.2430     14.9713      9.4679
Report dibuat:
- summary_by_stem: /Users/ameliacitra/Desktop/demucs_v4_musdb_project/outputs/reports/htdemucs_ft/test/summary_by_stem.csv
- track_scores: /Users/ameliacitra/Desktop/demucs_v4_musdb_project/outputs/reports/htdemucs_ft/test/track_scores.csv
- best_median_worst: /Users/ameliacitra/Desktop/demucs_v4_musdb_project/outputs/reports/htdemucs_ft/test/best_median_worst.csv
- automatic_findings: /Users/ameliacitra/Desktop/demucs_v4_musdb_project/outputs/reports/htdemucs_ft/test/automatic_findings.md