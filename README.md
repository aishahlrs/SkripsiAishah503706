# Analisis Pengaruh Pra-pemrosesan dan Ekstraksi Fitur terhadap Akurasi dan Konsistensi Deteksi *Sleep Apnea* Berbasis Radar IR-UWB

Repositori ini berisi seluruh kode sumber yang digunakan pada penelitian skripsi berjudul di atas. Penelitian mengklasifikasikan tiga kelas (Normal, Apnea, *Other event*) pada dataset **APNIWAVE** (radar IR-UWB) melalui ablasi sekuensial progresif, dengan fokus pada pengaruh keputusan pengolahan isyarat (*DC removal*, *bandpass filter*) dan rekayasa fitur (domain waktu, PSD, energi *wavelet*) terhadap akurasi **dan** konsistensi prediksi.

Klasifikator: *Random Forest* (100 pohon, `random_state=42`). Evaluasi: *stratified 5-fold cross-validation* (`shuffle=True`, `seed=42`).

---

## Dataset

Dataset **APNIWAVE** bersifat akses terbuka melalui Zenodo:

- **DOI:** [10.5281/zenodo.7703322](https://zenodo.org/records/7703322)

Unduh dua berkas berikut dan letakkan dalam satu folder:

| Berkas | Keterangan |
|---|---|
| `Labels Three Events.csv` | Label kelas per sampel, dimensi `(1011, 1)` |
| `Multiple Distances Three Events.csv` | Amplitudo radar, dimensi `(171870, 5)` = 1011 sampel × 170 titik waktu, 5 *range bin* |

> **PENTING — sesuaikan path dataset.** Skrip menggunakan path absolut, misalnya:
> ```python
> LABELS_PATH = '/Users/.../Datasets/APNIWAVE/Labels Three Events.csv'
> DIST_PATH   = '/Users/.../Datasets/APNIWAVE/Multiple Distances Three Events.csv'
> ```
> Ubah kedua konstanta ini di setiap skrip agar menunjuk ke lokasi dataset di komputer Anda.

---

## Lingkungan / Versi Pustaka

Seluruh angka pada Bab IV dihasilkan dengan versi berikut (identik di semua eksekusi):

| Komponen | Versi |
|---|---|
| Python | 3.14.3 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| Scikit-learn | 1.9.0 |
| PyWavelets | 1.8.0 |
| pandas | 3.0.3 |

Instalasi cepat:
```bash
pip install numpy scipy scikit-learn pywavelets pandas matplotlib
```

---

## Struktur Skrip

### Eksperimen utama (rantai ablasi)
| Skrip | Konfigurasi | Dimensi fitur |
|---|---|---|
| `exp1.py` | *Baseline* — sinyal mentah, 9 fitur domain waktu | 45 |
| `exp2.py` | *DC removal* + *bandpass filter* 0,1–0,5 Hz | 45 |
| `exp3.py` | *DC removal* + PSD + *wavelet* | 95 |
| `exp4.py` | Sinyal mentah + PSD + *wavelet* | 95 |

### Eksperimen dekomposisi komponen
| Skrip | Konfigurasi |
|---|---|
| `exp2_1.py` | Dekomposisi *preprocessing*: **DC removal saja** |
| `exp2_2.py` | Dekomposisi *preprocessing*: **BPF saja** |
| `exp3_1.py` | Dekomposisi fitur: **domain waktu + PSD** |
| `exp3_2.py` | Dekomposisi fitur: **domain waktu + *wavelet*** |

### Eksperimen pendukung
| Skrip | Konfigurasi |
|---|---|
| `exp1_temporal_normalized.py` | Normalisasi amplitudo *z-score* per jendela |
| `exp2_psd_not_normalized.py` | Fitur frekuensi murni (PSD-saja), spektrum mentah |
| `exp3_psd_normalized.py` | Fitur frekuensi murni (PSD-saja), spektrum ternormalisasi |

### Orkestrator
| Skrip | Fungsi |
|---|---|
| `run_all_experiments.py` | Menjalankan ketujuh konfigurasi (4 utama + 3 pendukung) dari satu sumber konsisten (protokol & *seed* identik). Menulis ringkasan ke `results/run_all_results.txt`. |

### Analisis pendukung
| Skrip | Keluaran |
|---|---|
| `check_separasi.py` | Angka separasi fitur *std* & entropi spektral per kelas (Q1/median/Q3, Mann–Whitney U, AUC, Cliff's delta) |
| `nyoba_stdentropi.py` | Aturan keputusan hierarkis **skema A** (*std* → entropi spektral) |
| `nyoba_entropistd.py` | Aturan keputusan hierarkis **skema B** (entropi spektral → *std*) |

### Visualisasi (gambar skripsi)
| Skrip | Gambar yang dihasilkan |
|---|---|
| `viz_karakterisasi.py` | Morfologi sinyal per kelas, PSD *mean*/median, *boxplot* fitur, ilustrasi bertahap *pipeline* |
| `viz_normalisasi.py` | *Confusion matrix z-score* & PSD-saja, perbandingan akurasi tujuh konfigurasi |

---

## Cara Menjalankan

1. Unduh dataset dan sesuaikan `LABELS_PATH` / `DIST_PATH` di skrip (lihat catatan di atas).
2. Reproduksi seluruh angka Bab IV sekaligus:
   ```bash
   python run_all_experiments.py
   ```
   Hasil ringkas tercetak ke layar dan tersimpan di `results/run_all_results.txt`.
3. Menjalankan satu eksperimen secara individual, misalnya:
   ```bash
   python exp4.py
   ```
4. Membuat ulang gambar dan tabel analisis:
   ```bash
   python viz_karakterisasi.py
   python viz_normalisasi.py
   python check_separasi.py
   python nyoba_stdentropi.py
   ```

---

## Sitasi

Jika menggunakan kode ini, mohon merujuk pada skripsi:

> Abqariyyah, A. L. N. (2026). *Analisis Pengaruh Pra-pemrosesan dan Ekstraksi Fitur terhadap Akurasi dan Konsistensi Deteksi Sleep Apnea Berbasis Radar IR-UWB*. Skripsi, Program Studi Teknik Biomedis, Universitas Gadjah Mada.

Dataset APNIWAVE:

> Uzunidis, D., dkk. (2023). *APNIWAVE dataset*. Zenodo. https://doi.org/10.5281/zenodo.7703322
