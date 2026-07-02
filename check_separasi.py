# -*- coding: utf-8 -*-
"""Hitung angka pasti separasi fitur std & entropi spektral (middle bin) per kelas.
Memakai definisi fitur yang IDENTIK dengan viz_karakterisasi.py."""
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import mannwhitneyu

LABELS_PATH = '/Users/aishahlaras/Desktop/scrispy/Datasets/APNIWAVE/Labels Three Events.csv'
DIST_PATH   = '/Users/aishahlaras/Desktop/scrispy/Datasets/APNIWAVE/Multiple Distances Three Events.csv'
FS, MID_BIN = 17.0, 2
CLASS_NAMES = ['Normal', 'Apnea', 'Other']

def dcr(s): s = s.astype(float); return s - s.mean()
def psd(s):
    f, Pxx = welch(dcr(s), fs=FS, nperseg=len(s)); return f, Pxx
def spectral_entropy(s):
    _, Pxx = psd(s); Pn = Pxx / (Pxx.sum() + 1e-12)
    return -np.sum(Pn * np.log2(Pn + 1e-12))

y = pd.read_csv(LABELS_PATH).values.ravel()
X = pd.read_csv(DIST_PATH).values.reshape(1011, 170, 5).transpose(0, 2, 1)

stds = {c: [] for c in CLASS_NAMES}; ents = {c: [] for c in CLASS_NAMES}
for i in range(len(X)):
    c = CLASS_NAMES[y[i]]
    stds[c].append(X[i, MID_BIN].std())
    ents[c].append(spectral_entropy(X[i, MID_BIN]))
stds = {c: np.array(v) for c, v in stds.items()}
ents = {c: np.array(v) for c, v in ents.items()}

print("Jumlah sampel per kelas:", {c: len(stds[c]) for c in CLASS_NAMES})

def ringkas(name, data):
    print(f"\n=== {name} : ringkasan per kelas (Q1 / median / Q3) ===")
    for c in CLASS_NAMES:
        q1, med, q3 = np.percentile(data[c], [25, 50, 75])
        print(f"  {c:7s}: Q1={q1:.4g}  median={med:.4g}  Q3={q3:.4g}  (n={len(data[c])})")

def auc_from_u(u, n1, n2): return u / (n1 * n2)

def pairwise(name, data):
    print(f"\n--- {name} : uji pasangan kelas (Mann-Whitney U, AUC, Cliff's delta) ---")
    for a, b in [('Normal','Apnea'), ('Normal','Other'), ('Apnea','Other')]:
        x, z = data[a], data[b]
        u, p = mannwhitneyu(x, z, alternative='two-sided')
        auc = auc_from_u(u, len(x), len(z))          # P(a > b)
        auc_sep = max(auc, 1 - auc)                   # kekuatan pemisahan (>=0.5)
        cliff = 2 * auc - 1
        tag = "SANGAT TERPISAH" if auc_sep >= 0.8 else ("terpisah sedang" if auc_sep >= 0.7 else "TUMPANG TINDIH")
        print(f"  {a:6s} vs {b:6s}: p={p:.2e}  AUC={auc_sep:.3f}  Cliff|d|={abs(cliff):.3f}  -> {tag}")

ringkas("STD", stds);  pairwise("STD", stds)
ringkas("ENTROPI SPEKTRAL", ents); pairwise("ENTROPI SPEKTRAL", ents)
