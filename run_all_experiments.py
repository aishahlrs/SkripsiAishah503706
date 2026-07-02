"""
run_all_experiments.py
======================
Menjalankan KETUJUH konfigurasi eksperimen skripsi dari satu sumber yang
konsisten, sehingga seluruh angka di Bab IV dapat dikunci dari satu run dengan
versi pustaka yang sama. Setiap eksperimen dimuat sebagai modul dari skrip
aslinya (build_feature_matrix masing-masing), lalu dievaluasi dengan protokol
yang IDENTIK: Stratified 5-fold CV (shuffle, seed=42) + Random Forest (100
pohon, seed=42).

Jalankan:  python run_all_experiments.py
Output  :  tabel ringkas ke layar + file results/run_all_results.txt
"""

import importlib.util
import os
import sys
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
import sklearn
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

HERE        = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ['Normal', 'Apnea', 'Other']
N_SPLITS    = 5
SEED        = 42

# (label tampilan, nama file skrip) — urutan sesuai narasi skripsi
EXPERIMENTS = [
    ("Exp 1  Baseline (raw, 9 waktu)",      "exp1.py"),
    ("Exp 2  DC removal + BPF",             "exp2.py"),
    ("Exp 3  DC removal + PSD + wavelet",   "exp3.py"),
    ("Exp 4  Raw + PSD + wavelet",          "exp4.py"),
    ("Pend.  Normalisasi z-score",          "exp1_temporal_normalized.py"),
    ("Pend.  PSD-only (Pxx mentah)",        "exp2_psd_not_normalized.py"),
    ("Pend.  PSD-only (Pxx ternormalisasi)","exp3_psd_normalized.py"),
]


def load_module(filename):
    """Muat sebuah skrip eksperimen sebagai modul tanpa menjalankan blok __main__."""
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(filename.replace('.py', ''), path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate(X_feat, y):
    """Protokol evaluasi seragam untuk semua eksperimen."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)

    accs, f1s, f1_cls = [], [], []
    cm_total = np.zeros((3, 3), dtype=int)
    for tr, te in skf.split(X_feat, y):
        clf.fit(X_feat[tr], y[tr])
        pred = clf.predict(X_feat[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average='macro'))
        f1_cls.append(f1_score(y[te], pred, average=None, labels=[0, 1, 2]))
        cm_total += confusion_matrix(y[te], pred, labels=[0, 1, 2])

    f1_cls = np.array(f1_cls)
    return {
        'acc_mean': np.mean(accs), 'acc_std': np.std(accs),
        'f1_mean':  np.mean(f1s),  'f1_std':  np.std(f1s),
        'f1_cls_mean': f1_cls.mean(axis=0), 'f1_cls_std': f1_cls.std(axis=0),
        'cm': cm_total,
    }


def versions_block():
    return (
        f"Tanggal run   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Python        : {platform.python_version()}\n"
        f"numpy         : {np.__version__}\n"
        f"scipy         : {scipy.__version__}\n"
        f"scikit-learn  : {sklearn.__version__}\n"
        f"PyWavelets    : {pywt.__version__}\n"
        f"pandas        : {pd.__version__}\n"
        f"Protokol      : Stratified {N_SPLITS}-fold CV (shuffle, seed={SEED}); "
        f"RandomForest(100, seed={SEED})\n"
    )


def main():
    out_lines = []

    def emit(line=""):
        print(line)
        out_lines.append(line)

    emit("=" * 78)
    emit("REPRODUKSI SELURUH EKSPERIMEN SKRIPSI — SATU SUMBER, SATU LINGKUNGAN")
    emit("=" * 78)
    emit(versions_block())

    # data dimuat sekali dari skrip pertama (load_data identik di semua skrip)
    base = load_module(EXPERIMENTS[0][1])
    X, y = base.load_data()
    emit(f"Data          : X={X.shape} (sampel x bin x titik), y={y.shape}, "
         f"kelas={sorted(set(y.tolist()))}\n")

    rows = []
    baseline_acc = None
    for label, fname in EXPERIMENTS:
        mod = load_module(fname)
        X_feat = mod.build_feature_matrix(X)
        res = evaluate(X_feat, y)
        if baseline_acc is None:
            baseline_acc = res['acc_mean']
        delta = (res['acc_mean'] - baseline_acc) * 100.0
        rows.append((label, X_feat.shape[1], res, delta))

        emit(f"[{fname}]")
        emit(f"  {label}")
        emit(f"  Fitur/sampel : {X_feat.shape[1]}")
        emit(f"  Akurasi      : {res['acc_mean']:.4f} ± {res['acc_std']:.4f}"
             f"   (Δ vs baseline: {delta:+.2f} pp)")
        emit(f"  Macro-F1     : {res['f1_mean']:.4f} ± {res['f1_std']:.4f}")
        emit("  F1 per kelas : " + "  ".join(
            f"{n}={m:.4f}±{s:.4f}"
            for n, m, s in zip(CLASS_NAMES, res['f1_cls_mean'], res['f1_cls_std'])))
        emit("")

    # ---- tabel ringkas ----
    emit("=" * 78)
    emit("TABEL RINGKAS")
    emit("=" * 78)
    emit(f"{'Konfigurasi':<38}{'Fitur':>6}{'Akurasi':>16}{'Macro-F1':>16}{'Δpp':>8}")
    emit("-" * 84)
    for label, nfeat, res, delta in rows:
        emit(f"{label:<38}{nfeat:>6}"
             f"{res['acc_mean']*100:>10.2f}±{res['acc_std']*100:<4.2f}"
             f"{res['f1_mean']*100:>10.2f}±{res['f1_std']*100:<4.2f}"
             f"{delta:>8.2f}")
    emit("=" * 78)

    # ---- simpan ----
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out_path = os.path.join(HERE, "results", "run_all_results.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nDisimpan ke: {out_path}")


if __name__ == "__main__":
    main()
