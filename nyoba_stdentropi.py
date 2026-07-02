# -*- coding: utf-8 -*-
"""Uji skema klasifikasi HIERARKIS 2 tahap (middle bin):
  Tahap 1 : std            -> {Normal,Apnea} vs Other
  Tahap 2 : entropi spektral -> Normal vs Apnea  (hanya untuk yang LOLOS tahap 1)
Tiap tahap = 1 ambang (decision stump). Ambang DICARI di data latih,
dievaluasi di data uji (Stratified 5-fold) supaya tidak overfit.
Definisi fitur identik dengan viz_karakterisasi.py.
"""
import numpy as np, pandas as pd
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score

LABELS_PATH = '/Users/aishahlaras/Desktop/scrispy/Datasets/APNIWAVE/Labels Three Events.csv'
DIST_PATH   = '/Users/aishahlaras/Desktop/scrispy/Datasets/APNIWAVE/Multiple Distances Three Events.csv'
FS, MID_BIN = 17.0, 2
NAMES = ['Normal', 'Apnea', 'Other']   # 0,1,2

def dcr(s): s = s.astype(float); return s - s.mean()
def spectral_entropy(s):
    _, P = welch(dcr(s), fs=FS, nperseg=len(s)); Pn = P/(P.sum()+1e-12)
    return -np.sum(Pn*np.log2(Pn+1e-12))

y = pd.read_csv(LABELS_PATH).values.ravel().astype(int)
X = pd.read_csv(DIST_PATH).values.reshape(1011,170,5).transpose(0,2,1)
std = np.array([X[i,MID_BIN].std() for i in range(len(X))])
ent = np.array([spectral_entropy(X[i,MID_BIN]) for i in range(len(X))])

def best_threshold(feat, pos_mask, hi_is_pos=True):
    """ambang yang memaksimalkan balanced-acc untuk biner pos vs neg."""
    cand = np.percentile(feat, np.linspace(1,99,99))
    best_t, best_s = cand[0], -1
    for t in cand:
        pred = (feat > t) if hi_is_pos else (feat < t)
        s = balanced_accuracy_score(pos_mask, pred)
        if s > best_s: best_s, best_t = s, t
    return best_t

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
CM = np.zeros((3,3), int)
for tr, te in skf.split(X, y):
    # --- Tahap 1: std tinggi = Other (kelas 2) ---
    t1 = best_threshold(std[tr], (y[tr]==2), hi_is_pos=True)
    # --- Tahap 2: dilatih pada Normal/Apnea sejati di train; entropi rendah = Normal ---
    na = tr[y[tr]!=2]
    t2 = best_threshold(ent[na], (y[na]==0), hi_is_pos=False)  # ent<t2 -> Normal
    # --- prediksi di test ---
    for i in te:
        if std[i] > t1:          pred = 2                 # Other
        else:                    pred = 0 if ent[i] < t2 else 1
        CM[y[i], pred] += 1

pred_all = []  # untuk metrik global rekonstruksi dari CM
acc = np.trace(CM)/CM.sum()
print("Ambang dicari per-fold (5-fold CV). Confusion matrix gabungan (baris=asli, kolom=prediksi):")
print("              pred:Normal  pred:Apnea  pred:Other")
for i,c in enumerate(NAMES):
    print(f"  asli {c:7s}: {CM[i,0]:10d} {CM[i,1]:11d} {CM[i,2]:11d}  (recall={CM[i,i]/CM[i].sum():.3f})")
print(f"\nAkurasi total      : {acc:.3f}")
# balanced acc & macro-F1 dari CM
recalls = [CM[i,i]/CM[i].sum() for i in range(3)]
print(f"Balanced accuracy  : {np.mean(recalls):.3f}")
precis = [CM[i,i]/CM[:,i].sum() if CM[:,i].sum()>0 else 0 for i in range(3)]
f1s = [2*p*r/(p+r) if (p+r)>0 else 0 for p,r in zip(precis,recalls)]
print(f"Macro-F1           : {np.mean(f1s):.3f}")
print(f"\nPer-kelas presisi  : " + "  ".join(f"{NAMES[i]}={precis[i]:.3f}" for i in range(3)))

# Diagnosis: berapa banyak Other yang bocor ke tahap-2, dan kemana kebocoran Normal/Apnea
print("\nDiagnosis kebocoran tahap-1 (Other yang salah dilewatkan ke tahap-2):",
      CM[2,0]+CM[2,1], "dari", CM[2].sum(), "sampel Other")
print("Bingung Normal<->Apnea di tahap-2:", CM[0,1], "Normal->Apnea ;", CM[1,0], "Apnea->Normal")
