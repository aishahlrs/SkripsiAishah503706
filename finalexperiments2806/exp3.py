import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

LABELS_PATH = '/Users/aishahlaras/Desktop/scrispy/Datasets/APNIWAVE/Labels Three Events.csv'
DIST_PATH   = '/Users/aishahlaras/Desktop/scrispy/Datasets/APNIWAVE/Multiple Distances Three Events.csv'
CLASS_NAMES = ['Normal', 'Apnea', 'Other']
FS          = 17.0
WAVELET     = 'db4'
WAV_LEVEL   = 4


def load_data():
    y = pd.read_csv(LABELS_PATH).values.ravel()
    X = pd.read_csv(DIST_PATH).values.reshape(1011, 170, 5).transpose(0, 2, 1)
    return X, y


def dc_remove(window):
    out = np.empty_like(window, dtype=float)
    for b in range(window.shape[0]):
        out[b] = window[b] - window[b].mean()
    return out


def psd_features(sig, fs=FS):
    f, Pxx = welch(sig, fs=fs, nperseg=len(sig))
    resp_mask  = (f >= 0.1) & (f <= 0.5)
    low_mask   = (f >  0.5) & (f <= 2.0)
    high_mask  = (f >  2.0)
    resp_power = Pxx[resp_mask].sum()
    low_power  = Pxx[low_mask].sum()
    high_power = Pxx[high_mask].sum()
    dom_freq   = f[resp_mask][np.argmax(Pxx[resp_mask])]
    Pxx_norm   = Pxx / (Pxx.sum() + 1e-12)
    sp_entropy = -np.sum(Pxx_norm * np.log2(Pxx_norm + 1e-12))
    return [resp_power, low_power, high_power, dom_freq, sp_entropy]


def wavelet_features(sig, wavelet=WAVELET, level=WAV_LEVEL):
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    return [np.sum(c ** 2) / len(c) for c in coeffs]


def extract_features(window):
    feats = []
    for b in range(window.shape[0]):
        s = window[b]
        feats += [
            np.mean(s), np.min(s), np.max(s), np.median(s),
            np.std(s), skew(s), kurtosis(s),
            np.percentile(s, 25), np.percentile(s, 75)
        ]
        feats += psd_features(s)
        feats += wavelet_features(s)
    return feats


def build_feature_matrix(X):
    return np.array([extract_features(dc_remove(X[i])) for i in range(len(X))])


def evaluate(X_feat, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    accs, f1s, f1_cls_all = [], [], []
    cm_total = np.zeros((3, 3), dtype=int)

    for fold, (tr, te) in enumerate(skf.split(X_feat, y)):
        clf.fit(X_feat[tr], y[tr])
        pred = clf.predict(X_feat[te])

        acc = accuracy_score(y[te], pred)
        f1  = f1_score(y[te], pred, average='macro')
        f1c = f1_score(y[te], pred, average=None, labels=[0, 1, 2])
        cm  = confusion_matrix(y[te], pred, labels=[0, 1, 2])

        accs.append(acc)
        f1s.append(f1)
        f1_cls_all.append(f1c)
        cm_total += cm

        print(f"Fold {fold+1}  Acc={acc:.4f}  MacroF1={f1:.4f}  "
              f"[Normal={f1c[0]:.4f}  Apnea={f1c[1]:.4f}  Other={f1c[2]:.4f}]")

    f1_cls_all = np.array(f1_cls_all)

    print(f"\n{'='*62}")
    print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"\nPer-class F1:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<8}: {f1_cls_all[:, i].mean():.4f} ± {f1_cls_all[:, i].std():.4f}")

    print(f"\nAggregated Confusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':14}" + "".join(f"{n:>10}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<12}" + "".join(f"{cm_total[i, j]:>10}" for j in range(3)))


if __name__ == "__main__":
    X, y = load_data()
    X_feat = build_feature_matrix(X)
    n_per_bin = 9 + 5 + (WAV_LEVEL + 1)
    print(f"Signal array  : {X.shape}   (samples × bins × timepoints)")
    print(f"Feature matrix: {X_feat.shape}  "
          f"(5 bins × {n_per_bin} features = {5 * n_per_bin} total)\n")
    evaluate(X_feat, y)