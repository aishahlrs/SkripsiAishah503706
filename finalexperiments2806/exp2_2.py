import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

LABELS_PATH = '/Users/aishahlaras/Desktop/scrispy/Datasets/APNIWAVE/Labels Three Events.csv'
DIST_PATH   = '/Users/aishahlaras/Desktop/scrispy/Datasets/APNIWAVE/Multiple Distances Three Events.csv'
CLASS_NAMES = ['Normal', 'Apnea', 'Other']
FS          = 17.0

def load_data():
    y = pd.read_csv(LABELS_PATH).values.ravel()
    X = pd.read_csv(DIST_PATH).values.reshape(1011, 170, 5).transpose(0, 2, 1)
    return X, y

def bandpass(sig, lo=0.1, hi=0.5, fs=FS, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, sig)

def preprocess(window):
    out = np.empty_like(window)
    for b in range(window.shape[0]):
        out[b] = bandpass(window[b])
    return out

def extract_features(window):
    feats = []
    for b in range(window.shape[0]):
        s = window[b]
        feats += [
            np.mean(s), np.min(s), np.max(s), np.median(s),
            np.std(s), skew(s), kurtosis(s),
            np.percentile(s, 25), np.percentile(s, 75)
        ]
    return feats

def build_feature_matrix(X):
    return np.array([extract_features(preprocess(X[i])) for i in range(len(X))])

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
    print(f"Signal array : {X.shape}   (samples × bins × timepoints)")
    print(f"Feature matrix: {X_feat.shape}  (samples × features)\n")
    evaluate(X_feat, y)
