# -*- coding: utf-8 -*-
"""Skema HIERARKIS ALTERNATIF (middle bin):
  Tahap 1 : entropi spektral -> Normal vs {Apnea,Other}
  Tahap 2 : std              -> Apnea vs Other  (hanya yang LOLOS tahap 1)
Bandingkan dengan skema lama (std dulu). CV 5-fold, ambang dicari di train.
"""
import numpy as np, pandas as pd
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

LABELS_PATH = '/Users/aishahlaras/Desktop/scrispy/Datasets/APNIWAVE/Labels Three Events.csv'
DIST_PATH   = '/Users/aishahlaras/Desktop/scrispy/Datasets/APNIWAVE/Multiple Distances Three Events.csv'
FS, MID_BIN = 17.0, 2
NAMES = ['Normal','Apnea','Other']

def dcr(s): s = s.astype(float); return s - s.mean()
def spectral_entropy(s):
    _, P = welch(dcr(s), fs=FS, nperseg=len(s)); Pn=P/(P.sum()+1e-12)
    return -np.sum(Pn*np.log2(Pn+1e-12))

y = pd.read_csv(LABELS_PATH).values.ravel().astype(int)
X = pd.read_csv(DIST_PATH).values.reshape(1011,170,5).transpose(0,2,1)
std = np.array([X[i,MID_BIN].std() for i in range(len(X))])
ent = np.array([spectral_entropy(X[i,MID_BIN]) for i in range(len(X))])

def best_threshold(feat, pos_mask, hi_is_pos=True):
    best_t,best_s = None,-1
    for t in np.percentile(feat, np.linspace(1,99,99)):
        pred = (feat>t) if hi_is_pos else (feat<t)
        s = balanced_accuracy_score(pos_mask, pred)
        if s>best_s: best_s,best_t = s,t
    return best_t

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
CM = np.zeros((3,3),int)
for tr,te in skf.split(X,y):
    # Tahap 1: entropi RENDAH = Normal (kelas 0)
    t1 = best_threshold(ent[tr], (y[tr]==0), hi_is_pos=False)
    # Tahap 2: dilatih pada Apnea/Other sejati; std RENDAH = Apnea (kelas1), std tinggi = Other(2)
    ao = tr[y[tr]!=0]
    t2 = best_threshold(std[ao], (y[ao]==1), hi_is_pos=False)  # std<t2 -> Apnea
    for i in te:
        if ent[i] < t1:  pred = 0                      # Normal
        else:            pred = 1 if std[i] < t2 else 2 # Apnea / Other
        CM[y[i],pred]+=1

acc = np.trace(CM)/CM.sum()
print("SKEMA BARU: entropi(Normal vs rest) -> std(Apnea vs Other)\n")
print("              pred:Normal  pred:Apnea  pred:Other")
for i,c in enumerate(NAMES):
    print(f"  asli {c:7s}: {CM[i,0]:10d} {CM[i,1]:11d} {CM[i,2]:11d}  (recall={CM[i,i]/CM[i].sum():.3f})")
recalls=[CM[i,i]/CM[i].sum() for i in range(3)]
precis=[CM[i,i]/CM[:,i].sum() if CM[:,i].sum()>0 else 0 for i in range(3)]
f1s=[2*p*r/(p+r) if (p+r)>0 else 0 for p,r in zip(precis,recalls)]
print(f"\nAkurasi total      : {acc:.3f}")
print(f"Balanced accuracy  : {np.mean(recalls):.3f}")
print(f"Macro-F1           : {np.mean(f1s):.3f}")
print(f"Per-kelas presisi  : " + "  ".join(f"{NAMES[i]}={precis[i]:.3f}" for i in range(3)))
print("\nDiagnosis:")
print("  Normal bocor ke tahap-2 (Normal->Apnea/Other):", CM[0,1]+CM[0,2], "dari", CM[0].sum())
print("  Bingung Apnea<->Other di tahap-2:", CM[1,2], "Apnea->Other ;", CM[2,1], "Other->Apnea")
