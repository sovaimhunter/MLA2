"""
把所有 ct_/t_ 配对特征全部差值化，测试是否提升准确率。
对比：原始特征 vs 部分 diff（已有）vs 全量 diff
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = "csgo_round_snapshots.csv"

WEAPON_PRICES = {
    "ak47":9000,"aug":3300,"awp":4750,"bizon":1400,"cz75auto":500,
    "elite":300,"famas":2050,"g3sg1":5000,"galilar":1800,"glock":200,
    "m249":5200,"m4a1s":2900,"m4a4":3100,"mac10":1050,"mag7":1300,
    "mp5sd":1500,"mp7":1500,"mp9":1250,"negev":1700,"nova":1050,
    "p90":2350,"r8revolver":600,"sawedoff":1100,"scar20":5000,
    "sg553":2750,"ssg08":1700,"ump45":1200,"xm1014":2000,
    "deagle":700,"fiveseven":500,"usps":200,"p250":300,
    "p2000":200,"tec9":500,
    "hegrenade":300,"flashbang":200,"smokegrenade":300,
    "incendiarygrenade":600,"molotovgrenade":400,"decoygrenade":50,
}

def make_round_ids(df):
    return (df["time_left"].diff().fillna(0) > 0).cumsum()

def add_all_diffs(df):
    """对所有 ct_xxx / t_xxx 配对列自动生成差值特征"""
    df = df.copy()
    ct_cols = [c for c in df.columns if c.startswith("ct_")]
    added = []
    for ct_col in ct_cols:
        t_col = "t_" + ct_col[3:]
        if t_col in df.columns:
            diff_col = "diff_" + ct_col[3:]
            df[diff_col] = df[ct_col] - df[t_col]
            added.append(diff_col)

    # 武器总价值
    ct_val = sum(df[f"ct_weapon_{w}"] * p for w, p in WEAPON_PRICES.items()
                 if f"ct_weapon_{w}" in df.columns)
    t_val  = sum(df[f"t_weapon_{w}"]  * p for w, p in WEAPON_PRICES.items()
                 if f"t_weapon_{w}"  in df.columns)
    df["ct_weapon_value"]   = ct_val
    df["t_weapon_value"]    = t_val
    df["diff_weapon_value"] = ct_val - t_val
    added += ["ct_weapon_value", "t_weapon_value", "diff_weapon_value"]

    return df, added

def evaluate(df, feature_cols, label):
    round_ids = make_round_ids(df)
    X = df[feature_cols].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df["round_winner"])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=round_ids))

    sc = StandardScaler()
    X_tr = sc.fit_transform(X[tr_idx])
    X_te = sc.transform(X[te_idx])

    model = RandomForestClassifier(n_estimators=500, max_depth=20,
                                   min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_tr, y[tr_idx])
    preds = model.predict(X_te)

    acc = accuracy_score(y[te_idx], preds)
    f1  = f1_score(y[te_idx], preds, average="weighted")
    print(f"  [{label}]  n_features={len(feature_cols):>3}  acc={acc:.4f}  f1={f1:.4f}")

    # top 10 feature importance
    imp = pd.Series(model.feature_importances_, index=feature_cols).nlargest(10)
    print("  Top 10:")
    for feat, v in imp.items():
        print(f"    {feat:<40} {v:.4f}")
    return acc, f1

def main():
    df = pd.read_csv(DATA_PATH)
    df["map"] = LabelEncoder().fit_transform(df["map"])
    df["bomb_planted"] = df["bomb_planted"].astype(int)

    df_diff, diff_cols = add_all_diffs(df)

    raw_cols      = [c for c in df.columns if c != "round_winner"]
    diff_only_cols = diff_cols + ["time_left", "bomb_planted", "map"]
    all_cols      = [c for c in df_diff.columns if c != "round_winner"]

    print("=== Feature Strategy Comparison ===\n")
    results = {}
    for label, cols, data in [
        ("Raw only",         raw_cols,       df),
        ("Diff only",        diff_only_cols, df_diff),
        ("Raw + All diffs",  all_cols,       df_diff),
    ]:
        acc, f1 = evaluate(data, cols, label)
        results[label] = (acc, f1)
        print()

    print("=== Summary ===")
    print(f"  {'Strategy':<20} {'Accuracy':>9}  {'F1':>7}")
    print("  " + "-"*42)
    for label, (acc, f1) in results.items():
        print(f"  {label:<20} {acc:>9.4f}  {f1:>7.4f}")

if __name__ == "__main__":
    main()
