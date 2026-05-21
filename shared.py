"""
Shared utilities for CS:GO round winner prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = "csgo_round_snapshots.csv"

WEAPON_PRICES = {
    "ak47": 9000, "aug": 3300, "awp": 4750, "bizon": 1400, "cz75auto": 500,
    "elite": 300, "famas": 2050, "g3sg1": 5000, "galilar": 1800, "glock": 200,
    "m249": 5200, "m4a1s": 2900, "m4a4": 3100, "mac10": 1050, "mag7": 1300,
    "mp5sd": 1500, "mp7": 1500, "mp9": 1250, "negev": 1700, "nova": 1050,
    "p90": 2350, "r8revolver": 600, "sawedoff": 1100, "scar20": 5000,
    "sg553": 2750, "ssg08": 1700, "ump45": 1200, "xm1014": 2000,
    "deagle": 700, "fiveseven": 500, "usps": 200, "p250": 300,
    "p2000": 200, "tec9": 500,
    "hegrenade": 300, "flashbang": 200, "smokegrenade": 300,
    "incendiarygrenade": 600, "molotovgrenade": 400, "decoygrenade": 50,
}

ECONOMIC_KEYWORDS = ["money", "helmet", "defuse", "weapon", "grenade", "score"]
SURVIVAL_KEYWORDS = ["health", "armor", "players_alive", "bomb_planted", "time_left"]

_cache = {}


def make_round_ids(df):
    return (df["time_left"].diff().fillna(0) > 0).cumsum()


def load_base():
    """Load and minimally preprocess the full DataFrame (used by RQ2/RQ3/figures)."""
    df = pd.read_csv(DATA_PATH)
    df["map"] = LabelEncoder().fit_transform(df["map"])
    df["bomb_planted"] = df["bomb_planted"].astype(int)
    return df


def load_data(test_size=0.2, scale=True, return_groups=False):
    """Load, split, and optionally scale data; results are cached (used by RQ1)."""
    key = (test_size, scale)
    if key in _cache:
        result = _cache[key]
        return result if return_groups else result[:4]

    df = pd.read_csv(DATA_PATH)
    df["map"] = LabelEncoder().fit_transform(df["map"])
    df["bomb_planted"] = df["bomb_planted"].astype(int)

    round_ids = make_round_ids(df)
    print(f"Total rounds: {round_ids.nunique()}")

    X = df.drop(columns=["round_winner"]).values.astype(np.float32)
    y = LabelEncoder().fit_transform(df["round_winner"])

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=round_ids))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_round_ids = round_ids.values[train_idx]

    if scale:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test  = sc.transform(X_test)

    _cache[key] = (X_train, X_test, y_train, y_test, train_round_ids)
    return (X_train, X_test, y_train, y_test, train_round_ids) if return_groups else (X_train, X_test, y_train, y_test)


def engineer_features(df):
    df = df.copy()
    df["health_diff"]        = df["ct_health"]       - df["t_health"]
    df["armor_diff"]         = df["ct_armor"]         - df["t_armor"]
    df["players_alive_diff"] = df["ct_players_alive"] - df["t_players_alive"]
    df["money_diff"]         = df["ct_money"]         - df["t_money"]
    df["helmets_diff"]       = df["ct_helmets"]       - df["t_helmets"]
    df["defuse_kits"]        = df["ct_defuse_kits"]

    ct_val = sum(df[f"ct_weapon_{w}"] * p for w, p in WEAPON_PRICES.items()
                 if f"ct_weapon_{w}" in df.columns)
    t_val  = sum(df[f"t_weapon_{w}"]  * p for w, p in WEAPON_PRICES.items()
                 if f"t_weapon_{w}"  in df.columns)
    df["ct_weapon_value"]   = ct_val
    df["t_weapon_value"]    = t_val
    df["weapon_value_diff"] = ct_val - t_val

    grenade_cols_ct = [c for c in df.columns if c.startswith("ct_grenade_")]
    grenade_cols_t  = [c for c in df.columns if c.startswith("t_grenade_")]
    df["ct_total_grenades"] = df[grenade_cols_ct].sum(axis=1)
    df["t_total_grenades"]  = df[grenade_cols_t].sum(axis=1)
    df["grenade_diff"]      = df["ct_total_grenades"] - df["t_total_grenades"]
    return df


def add_all_diffs(df):
    df = df.copy()
    ct_cols = [c for c in df.columns if c.startswith("ct_")]
    added = []
    for ct_col in ct_cols:
        t_col = "t_" + ct_col[3:]
        if t_col in df.columns:
            diff_col = "diff_" + ct_col[3:]
            df[diff_col] = df[ct_col] - df[t_col]
            added.append(diff_col)

    ct_val = sum(df[f"ct_weapon_{w}"] * p for w, p in WEAPON_PRICES.items()
                 if f"ct_weapon_{w}" in df.columns)
    t_val  = sum(df[f"t_weapon_{w}"]  * p for w, p in WEAPON_PRICES.items()
                 if f"t_weapon_{w}"  in df.columns)
    df["ct_weapon_value"]   = ct_val
    df["t_weapon_value"]    = t_val
    df["diff_weapon_value"] = ct_val - t_val
    added += ["ct_weapon_value", "t_weapon_value", "diff_weapon_value"]
    return df, added


def split_fit_eval(df, feature_cols, mask=None, n_estimators=500, max_depth=20, min_samples_split=5):
    if mask is not None:
        seg = df[mask].copy()
        round_ids = make_round_ids(df)[mask]
    else:
        seg = df.copy()
        round_ids = make_round_ids(df)

    X = seg[feature_cols].values.astype(np.float32)
    y = LabelEncoder().fit_transform(seg["round_winner"])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=round_ids))

    sc = StandardScaler()
    X_tr = sc.fit_transform(X[tr_idx])
    X_te = sc.transform(X[te_idx])
    y_tr, y_te = y[tr_idx], y[te_idx]

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return accuracy_score(y_te, preds), f1_score(y_te, preds, average="weighted"), model, feature_cols, y_te, preds


def categorise(feat):
    for k in ECONOMIC_KEYWORDS:
        if k in feat:
            return "economic"
    for k in SURVIVAL_KEYWORDS:
        if k in feat:
            return "survival"
    return "other"
