# -*- coding: utf-8 -*-
"""
Visualisasi hasil 7 konfigurasi (4 inti + z-score + 2 PSD-saja).
Menghasilkan 3 gambar:
  1. perbandingan_7config.png  -> bar chart akurasi (mean +- std) ke-7 konfigurasi
  2. cm_zscore.png             -> confusion matrix eksperimen z-score
  3. cm_psdonly.png            -> confusion matrix eksperimen PSD-saja (Pxx mentah)

Jalankan:  ./.venv/bin/python viz_normalisasi.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt, welch
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS_PATH = '/Users/aishahlaras/Desktop/scrispy/Datasets/APNIWAVE/Labels Three Events.csv'
DIST_PATH   = '/Users/aishahlaras/Desktop/scrispy/Datasets/APNIWAVE/Multiple Distances Three Events.csv'
CLASS_NAMES = ['Normal', 'Apnea', 'Other']
FS = 17.0

# Output: simpan ke folder figures skripsi DAN folder eksperimen
THESIS_FIG = '/Users/aishahlaras/Desktop/Skripsi_Aishah_Laras2/contents/chapter-4/figures'
LOCAL_FIG  = '/Users/aishahlaras/Desktop/scrispy/nyobaAPNIWAVE/4_6_2026/results'


def load_data():
    y = pd.read_csv(LABELS_PATH).values.ravel()
    X = pd.read_csv(DIST_PATH).values.reshape(1011, 170, 5).transpose(0, 2, 1)
    return X, y


# ---------- blok fitur ----------
def temporal(s):
    return [np.mean(s), np.min(s), np.max(s), np.median(s), np.std(s),
            skew(s), kurtosis(s), np.percentile(s, 25), np.percentile(s, 75)]

def psd_feats(s, normalize=False):
    # Mirror persis skrip asli: exp3 (normalize=True) menormalisasi Pxx lalu
    # menghitung entropi dari Pxx ternormalisasi; exp2 (normalize=False)
    # memakai Pxx mentah untuk daya pita & entropi dari Pxx/sum terpisah.
    s = s.astype(float)
    s = s - s.mean()
    f, Pxx = welch(s, fs=FS, nperseg=len(s))
    resp_mask = (f >= 0.1) & (f <= 0.5)
    if normalize:
        Pxx = Pxx / (Pxx.sum() + 1e-12)
        resp = Pxx[resp_mask].sum()
        low  = Pxx[(f > 0.5) & (f <= 2.0)].sum()
        high = Pxx[(f > 2.0)].sum()
        dom  = f[resp_mask][np.argmax(Pxx[resp_mask])]
        ent  = -np.sum(Pxx * np.log2(Pxx + 1e-12))
    else:
        resp = Pxx[resp_mask].sum()
        low  = Pxx[(f > 0.5) & (f <= 2.0)].sum()
        high = Pxx[(f > 2.0)].sum()
        dom  = f[resp_mask][np.argmax(Pxx[resp_mask])]
        Pn   = Pxx / (Pxx.sum() + 1e-12)
        ent  = -np.sum(Pn * np.log2(Pn + 1e-12))
    return [resp, low, high, dom, ent]

def wavelet_feats(s):
    return [np.sum(c**2)/len(c) for c in pywt.wavedec(s, 'db4', level=4)]

def bandpass(s, lo=0.1, hi=0.5, order=3):
    b, a = butter(order, [lo/(FS/2), hi/(FS/2)], btype='band')
    return filtfilt(b, a, s)

def zscore(s):
    s = s.astype(float); sd = s.std()
    return (s - s.mean())/sd if sd > 1e-12 else s - s.mean()


# ---------- 7 konfigurasi: window (5x170) -> vektor fitur ----------
def f_baseline(w):  return [v for b in w for v in temporal(b)]
def f_dcbpf(w):     return [v for b in w for v in temporal(bandpass(b - b.mean()))]
def f_dcfreq(w):    return [v for b in w for v in temporal(b - b.mean()) + psd_feats(b - b.mean()) + wavelet_feats(b - b.mean())]
def f_rawfreq(w):   return [v for b in w for v in temporal(b) + psd_feats(b) + wavelet_feats(b)]
def f_zscore(w):    return [v for b in w for v in temporal(zscore(b))]
def f_psd_raw(w):   return [v for b in w for v in psd_feats(b, normalize=False)]
def f_psd_norm(w):  return [v for b in w for v in psd_feats(b, normalize=True)]

CONFIGS = [
    ("Exp 1\nBaseline",      f_baseline),
    ("Exp 2\nDC+BPF",        f_dcbpf),
    ("Exp 3\nDC+Freq",       f_dcfreq),
    ("Exp 4\nRaw+Freq",      f_rawfreq),
    ("Z-score\n(temporal)",  f_zscore),
    ("PSD-saja\n(mentah)",   f_psd_raw),
    ("PSD-saja\n(ternorm.)", f_psd_norm),
]


def evaluate(build, X, y):
    feat = np.array([build(X[i]) for i in range(len(X))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    accs = []
    for tr, te in skf.split(feat, y):
        clf.fit(feat[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(feat[te])))
    pred = cross_val_predict(clf, feat, y, cv=skf)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    return np.mean(accs), np.std(accs), cm


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
    ax.set_title(title, fontsize=11, fontweight='bold')
    rowsum = cm.sum(axis=1, keepdims=True)
    for i in range(3):
        for j in range(3):
            pct = 100*cm[i, j]/rowsum[i, 0]
            ax.text(j, i, f"{cm[i,j]}\n({pct:.1f}%)", ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()*0.5 else 'black', fontsize=10)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for d in (THESIS_FIG, LOCAL_FIG):
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, path), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print("  tersimpan:", path)


def main():
    print("Memuat data & menjalankan 7 konfigurasi (mohon tunggu ~1-2 menit)...")
    X, y = load_data()
    results = {}
    cms = {}
    for label, build in CONFIGS:
        m, sd, cm = evaluate(build, X, y)
        key = label.replace("\n", " ")
        results[key] = (m, sd)
        cms[key] = cm
        print(f"  {key:22s} acc = {m*100:5.2f}% +/- {sd*100:.2f}%")

    # --- 1. bar chart perbandingan 7 konfigurasi ---
    labels = [c[0] for c in CONFIGS]
    means = [results[c[0].replace(chr(10), ' ')][0]*100 for c in CONFIGS]
    stds  = [results[c[0].replace(chr(10), ' ')][1]*100 for c in CONFIGS]
    colors = ['#2E75B6', '#C0504D', '#2E75B6', '#2E75B6', '#C0504D', '#ED9B40', '#ED9B40']
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(range(7), means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.6)
    ax.axhline(means[0], ls='--', color='gray', lw=1, label='Baseline (91,29%)')
    ax.set_xticks(range(7)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Akurasi (%)'); ax.set_ylim(60, 96)
    ax.set_title('Perbandingan Akurasi 7 Konfigurasi (mean ± std, 5-fold CV)', fontweight='bold')
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.6, f"{m:.1f}", ha='center', fontsize=9, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    for d in (THESIS_FIG, LOCAL_FIG):
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, 'perbandingan_7config.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print("  tersimpan: perbandingan_7config.png")

    # --- 2 & 3. confusion matrix z-score & PSD-saja ---
    plot_cm(cms["Z-score (temporal)"],
            "Confusion Matrix — Z-score (akurasi 72,2%)", 'cm_zscore.png')
    plot_cm(cms["PSD-saja (mentah)"],
            "Confusion Matrix — PSD-saja (akurasi 78,3%)", 'cm_psdonly.png')

    print("\nSelesai. Gambar tersimpan di:")
    print("  -", THESIS_FIG)
    print("  -", LOCAL_FIG)


if __name__ == "__main__":
    main()
