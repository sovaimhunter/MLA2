"""
验证假设：早段（>120s）高准确率来自经济惯性，而非战术信息。
方法：分别在早/中/晚段提取 RF 特征重要性，看经济特征占比是否在早段最高。
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = "csgo_round_snapshots.csv"

ECONOMIC_KEYWORDS = ["money", "helmet", "defuse", "weapon", "grenade", "score"]
SURVIVAL_KEYWORDS = ["health", "armor", "players_alive", "bomb_planted", "time_left"]

def make_round_ids(df):
    return (df["time_left"].diff().fillna(0) > 0).cumsum()

def categorise(feat):
    for k in ECONOMIC_KEYWORDS:
        if k in feat:
            return "economic"
    for k in SURVIVAL_KEYWORDS:
        if k in feat:
            return "survival"
    return "other"

def fit_and_importance(df, mask, label):
    seg = df[mask].copy()
    round_ids = make_round_ids(df)[mask]
    feature_cols = [c for c in df.columns if c != "round_winner"]

    X = seg[feature_cols].values.astype(np.float32)
    y = LabelEncoder().fit_transform(seg["round_winner"])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=round_ids))

    sc = StandardScaler()
    X_tr = sc.fit_transform(X[tr_idx])
    X_te = sc.transform(X[te_idx])

    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_tr, y[tr_idx])
    acc = model.score(X_te, y[te_idx])

    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp_df = imp.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df["category"] = imp_df["feature"].apply(categorise)

    cat_share = imp_df.groupby("category")["importance"].sum()
    top10 = imp_df.nlargest(10, "importance")

    print(f"\n{'='*50}")
    print(f"Segment: {label}  (n={mask.sum()}, acc={acc:.4f})")
    print(f"{'='*50}")
    print("Category importance share:")
    for cat, share in cat_share.sort_values(ascending=False).items():
        print(f"  {cat:<12} {share:.4f}  ({share*100:.1f}%)")
    print("\nTop 10 features:")
    for _, row in top10.iterrows():
        print(f"  {row['feature']:<35} {row['importance']:.4f}  [{row['category']}]")

    return acc, cat_share

def main():
    df = pd.read_csv(DATA_PATH)
    df["map"] = LabelEncoder().fit_transform(df["map"])
    df["bomb_planted"] = df["bomb_planted"].astype(int)

    early = df["time_left"] > 120
    mid   = (df["time_left"] >= 60) & (df["time_left"] <= 120)
    late  = df["time_left"] < 60

    results = {}
    for label, mask in [("Early >120s", early), ("Mid 60-120s", mid), ("Late <60s", late)]:
        acc, cat_share = fit_and_importance(df, mask, label)
        results[label] = {"acc": acc, "cat_share": cat_share}

    print("\n\n=== Economic vs Survival share across segments ===")
    print(f"{'Segment':<15} {'Acc':>6}  {'Economic':>10}  {'Survival':>10}")
    print("-" * 50)
    for label, v in results.items():
        eco = v["cat_share"].get("economic", 0)
        sur = v["cat_share"].get("survival", 0)
        print(f"{label:<15} {v['acc']:>6.4f}  {eco:>10.4f}  {sur:>10.4f}")

if __name__ == "__main__":
    main()
