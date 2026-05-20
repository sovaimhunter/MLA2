import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = "csgo_round_snapshots.csv"

_cache = {}


def _make_round_ids(df):
    # 每当 time_left 比上一行大，说明新 round 开始了
    new_round = df["time_left"].diff().fillna(0) > 0
    return new_round.cumsum()


def load_data(test_size=0.2, scale=True):
    key = (test_size, scale)
    if key in _cache:
        return _cache[key]

    df = pd.read_csv(DATA_PATH)

    df["map"] = LabelEncoder().fit_transform(df["map"])
    df["bomb_planted"] = df["bomb_planted"].astype(int)

    round_ids = _make_round_ids(df)
    print(f"Total rounds: {round_ids.nunique()}")

    X = df.drop(columns=["round_winner"]).values.astype(np.float32)
    y = LabelEncoder().fit_transform(df["round_winner"])  # CT=0, T=1

    # 按 round 整体划分，防止同一 round 的快照泄露到 train/test 两侧
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=round_ids))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    _cache[key] = (X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test
