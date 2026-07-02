# -*- coding: utf-8 -*-
"""
Regenerasi REPRODUCIBLE gambar karakterisasi Bab IV (Subbab 4.1 & 4.2):
  1. fig41_signal_morfologi.png  -> 1 sampel representatif tiap kelas (DC-removed), sumbu-y terpisah
  2. fig42_signal_shared.png     -> sampel yang sama, sumbu-y disamakan
  3. fig43_psd_mean_median.png   -> PSD per kelas (log), panel (a) mean & (b) median
  4. feat_boxplot.png            -> boxplot std (log) & entropi spektral per kelas
  5. contoh_bertahap.png         -> 1 jendela Normal melalui tahap pipeline: raw, DC-removed, BPF, PSD

Sifat reproducible:
  - Pemilihan "sampel representatif" deterministik = sampel yang std middle-bin-nya
    paling dekat ke MEDIAN kelas (indeks dicetak ke layar).
  - Semua angka kunci dicetak agar bisa dicocokkan dengan narasi skripsi.

Jalankan:  ./.venv/bin/python viz_karakterisasi.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ konfigurasi
LABELS_PATH = '/Users/aishahlaras/Desktop/scrispy/Datasets/APNIWAVE/Labels Three Events.csv'
DIST_PATH   = '/Users/aishahlaras/Desktop/scrispy/Datasets/APNIWAVE/Multiple Distances Three Events.csv'
FS          = 17.0
MID_BIN     = 2          # middle range bin (0..4)
CLASS_NAMES = ['Normal', 'Apnea', 'Other']
COL         = {'Normal': '#2E7D4F', 'Apnea': '#ED9B40', 'Other': '#C0504D'}

OUTDIR = '/Users/aishahlaras/Desktop/scrispy/nyobaAPNIWAVE/4_6_2026/results/karakterisasi_repro'
# Set True kalau ingin sekaligus menimpa figur di folder skripsi:
SAVE_TO_THESIS = False
THESIS_FIG = '/Users/aishahlaras/Desktop/Skripsi_Aishah_Laras3/contents/chapter-4/figures'
os.makedirs(OUTDIR, exist_ok=True)


# ------------------------------------------------------------------ util
def load_data():
    y = pd.read_csv(LABELS_PATH).values.ravel()
    X = pd.read_csv(DIST_PATH).values.reshape(1011, 170, 5).transpose(0, 2, 1)
    return X, y

def dcr(s):                       # DC removal
    s = s.astype(float); return s - s.mean()

def bandpass(s, lo=0.1, hi=0.5, order=3):
    b, a = butter(order, [lo/(FS/2), hi/(FS/2)], btype='band')
    return filtfilt(b, a, s)

def psd(s):                       # Welch PSD pada sinyal DC-removed
    f, Pxx = welch(dcr(s), fs=FS, nperseg=len(s))
    return f, Pxx

def spectral_entropy(s):
    _, Pxx = psd(s)
    Pn = Pxx / (Pxx.sum() + 1e-12)
    return -np.sum(Pn * np.log2(Pn + 1e-12))

def resp_ratio(s):                # proporsi daya di pita napas 0,1-0,5 Hz
    f, Pxx = psd(s)
    return Pxx[(f >= 0.1) & (f <= 0.5)].sum() / (Pxx.sum() + 1e-12)

def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, name), dpi=160, bbox_inches='tight')
    if SAVE_TO_THESIS:
        os.makedirs(THESIS_FIG, exist_ok=True)
        fig.savefig(os.path.join(THESIS_FIG, name), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print("   tersimpan:", name)


# ------------------------------------------------------------------ pemilihan sampel representatif
def representative_index(X, y, cls):
    """Sampel yang std middle-bin-nya paling dekat ke median kelas (deterministik)."""
    idx = np.where(y == cls)[0]
    stds = np.array([X[i, MID_BIN].std() for i in idx])
    pick = idx[np.argmin(np.abs(stds - np.median(stds)))]
    return pick


# ------------------------------------------------------------------ FIGUR
def fig_morfologi(X, reps, shared=False):
    t = np.arange(170) / FS
    sigs = {c: dcr(X[reps[c], MID_BIN]) for c in CLASS_NAMES}
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    if shared:
        lo = min(s.min() for s in sigs.values()); hi = max(s.max() for s in sigs.values())
        pad = 0.05 * (hi - lo)
    for ax, c in zip(axes, CLASS_NAMES):
        ax.plot(t, sigs[c], color=COL[c], lw=1.4)
        ax.set_title(f"{c} (sampel #{reps[c]})", fontweight='bold', color=COL[c])
        ax.set_xlabel("Waktu (detik)")
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        if shared: ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Amplitudo (DC-removed)")
    fig.tight_layout()
    save(fig, "fig42_signal_shared.png" if shared else "fig41_signal_morfologi.png")


def fig_psd(X, y):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6))
    # median = daya PITA napas (0,1-0,5 Hz);  rasio mean/median = daya TOTAL (sesuai teks skripsi)
    print("\n[PSD] median = daya pita napas 0,1-0,5 Hz ; rasio mean/median = daya TOTAL:")
    for ci, c in enumerate(CLASS_NAMES):
        idx = np.where(y == ci)[0]
        spec = []; bandpow = []; totpow = []
        for i in idx:
            f, Pxx = psd(X[i, MID_BIN]); spec.append(Pxx)
            bandpow.append(Pxx[(f >= 0.1) & (f <= 0.5)].sum())
            totpow.append(Pxx.sum())
        spec = np.array(spec); bandpow = np.array(bandpow); totpow = np.array(totpow)
        mean_s = spec.mean(axis=0); med_s = np.median(spec, axis=0)
        axes[0].semilogy(f, mean_s, color=COL[c], lw=1.6, label=c)
        axes[1].semilogy(f, med_s, color=COL[c], lw=1.6, label=c)
        ratio = totpow.mean() / (np.median(totpow) + 1e-30)
        print(f"   {c:7s}: median_pita={np.median(bandpow):.3e}   rasio_total mean/median={ratio:.1f}x")
    for ax, ttl in zip(axes, ["(a) Rata-rata (mean)", "(b) Median"]):
        ax.axvspan(0.1, 0.5, color='gray', alpha=0.15)
        ax.set_title(ttl, fontweight='bold'); ax.set_xlabel("Frekuensi (Hz)")
        ax.set_xlim(0, FS/2); ax.legend(); ax.grid(alpha=0.25, which='both')
    axes[0].set_ylabel("PSD (skala log)")
    fig.tight_layout()
    save(fig, "fig43_psd_mean_median.png")


def fig_boxplot(X, y):
    stds = {c: [] for c in CLASS_NAMES}; ents = {c: [] for c in CLASS_NAMES}
    for i in range(len(X)):
        c = CLASS_NAMES[y[i]]
        stds[c].append(X[i, MID_BIN].std())
        ents[c].append(spectral_entropy(X[i, MID_BIN]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    colors = [COL[c] for c in CLASS_NAMES]
    bp1 = axes[0].boxplot([stds[c] for c in CLASS_NAMES], labels=CLASS_NAMES, patch_artist=True)
    axes[0].set_yscale('log'); axes[0].set_title("Distribusi std (skala log)", fontweight='bold')
    axes[0].set_ylabel("std (middle bin)")
    bp2 = axes[1].boxplot([ents[c] for c in CLASS_NAMES], labels=CLASS_NAMES, patch_artist=True)
    axes[1].set_title("Distribusi entropi spektral", fontweight='bold'); axes[1].set_ylabel("entropi (bit)")
    for bp in (bp1, bp2):
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        for med in bp['medians']: med.set_color('black')
    for ax in axes: ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "feat_boxplot.png")
    print("\n[BOXPLOT] median per kelas:")
    for c in CLASS_NAMES:
        print(f"   {c:7s}: std={np.median(stds[c]):.6f}   entropi={np.median(ents[c]):.2f}")


def fig_bertahap(X, reps):
    i = reps['Normal']; raw = X[i, MID_BIN].astype(float); t = np.arange(170) / FS
    d = dcr(raw); bp = bandpass(d); f, Pxx = welch(d, fs=FS, nperseg=len(d))
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 6.6))
    ax[0, 0].plot(t, raw, color='#888'); ax[0, 0].set_title("(a) Sinyal mentah (ada offset DC)", fontweight='bold')
    ax[0, 1].plot(t, d, color='#1C7293'); ax[0, 1].set_title("(b) Setelah DC removal", fontweight='bold')
    ax[1, 0].plot(t, bp, color='#2E7D4F'); ax[1, 0].set_title("(c) Setelah bandpass 0,1-0,5 Hz", fontweight='bold')
    ax[1, 1].semilogy(f, Pxx, color='#C0504D'); ax[1, 1].axvspan(0.1, 0.5, color='gray', alpha=0.15)
    ax[1, 1].set_title("(d) PSD (Welch)", fontweight='bold'); ax[1, 1].set_xlabel("Frekuensi (Hz)"); ax[1, 1].set_xlim(0, FS/2)
    for a in [ax[0, 0], ax[0, 1], ax[1, 0]]:
        a.set_xlabel("Waktu (detik)"); a.axhline(0, color='gray', lw=0.5, ls='--'); a.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "contoh_bertahap.png")
    dom = f[(f >= 0.1) & (f <= 0.5)][np.argmax(Pxx[(f >= 0.1) & (f <= 0.5)])]
    print(f"\n[BERTAHAP] jendela Normal #{i}:")
    print(f"   mean={raw.mean():.3e}  min={raw.min():.3e}  max={raw.max():.3e}  std={raw.std():.3e}")
    print(f"   dom_freq={dom:.2f} Hz   resp_ratio={resp_ratio(raw):.3f}   entropi={spectral_entropy(raw):.3f}")


# ------------------------------------------------------------------ main
def main():
    print("Memuat data...")
    X, y = load_data()
    reps = {c: representative_index(X, y, i) for i, c in enumerate(CLASS_NAMES)}
    print("Sampel representatif (deterministik, std~median kelas):", reps)
    print("\nMembuat gambar ->", OUTDIR)
    fig_morfologi(X, reps, shared=False)
    fig_morfologi(X, reps, shared=True)
    fig_psd(X, y)
    fig_boxplot(X, y)
    fig_bertahap(X, reps)
    print("\nSelesai. Bandingkan dengan figur lama sebelum menimpa.")
    print("Set SAVE_TO_THESIS=True di skrip untuk sekaligus menyimpan ke folder skripsi.")


if __name__ == "__main__":
    main()
