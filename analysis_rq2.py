import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_PATH = "csgo_round_snapshots.csv"

def make_round_ids(df):
    return (df["time_left"].diff().fillna(0) > 0).cumsum()

def prepare_segment(df, mask):
    seg = df[mask].copy()
    round_ids = make_round_ids(df)[mask]

    X = seg.drop(columns=["round_winner"]).values.astype(np.float32)
    y = LabelEncoder().fit_transform(seg["round_winner"])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=round_ids))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def run_segment(name, X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="weighted")
    print(f"\n[{name}]  n_test={len(y_test)}")
    print(f"  Accuracy: {acc:.4f}  F1: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["CT","T"]))
    return acc, f1

def main():
    df = pd.read_csv(DATA_PATH)
    df["map"] = LabelEncoder().fit_transform(df["map"])
    df["bomb_planted"] = df["bomb_planted"].astype(int)

    early = df["time_left"] > 120
    mid   = (df["time_left"] >= 60) & (df["time_left"] <= 120)
    late  = df["time_left"] < 60

    print("Segment sizes:")
    print(f"  Early (>120s): {early.sum()}")
    print(f"  Mid (60-120s): {mid.sum()}")
    print(f"  Late (<60s):   {late.sum()}")

    results = {}
    for name, mask in [("Early >120s", early), ("Mid 60-120s", mid), ("Late <60s", late)]:
        X_tr, X_te, y_tr, y_te = prepare_segment(df, mask)
        acc, f1 = run_segment(name, X_tr, X_te, y_tr, y_te)
        results[name] = {"accuracy": acc, "f1": f1, "n": mask.sum()}

    # bomb planted effect per segment
    print("\n=== Bomb Planted Rate per Segment ===")
    for name, mask in [("Early", early), ("Mid", mid), ("Late", late)]:
        rate = df[mask]["bomb_planted"].mean()
        print(f"  {name}: {rate:.3f}")

    print("\n=== Summary ===")
    for name, v in results.items():
        print(f"  {name}: acc={v['accuracy']:.4f}  f1={v['f1']:.4f}  n={v['n']}")

if __name__ == "__main__":
    main()
