import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = "csgo_round_snapshots.csv"

# CS:GO 武器价格表（近似值，单位：$）
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

def engineer_features(df):
    df = df.copy()

    # 生存类特征差值
    df["health_diff"]        = df["ct_health"]        - df["t_health"]
    df["armor_diff"]         = df["ct_armor"]          - df["t_armor"]
    df["players_alive_diff"] = df["ct_players_alive"]  - df["t_players_alive"]

    # 经济类特征差值
    df["money_diff"]         = df["ct_money"]          - df["t_money"]
    df["helmets_diff"]       = df["ct_helmets"]        - df["t_helmets"]
    df["defuse_kits"]        = df["ct_defuse_kits"]

    # 武器总价值
    ct_val = sum(df[f"ct_weapon_{w}"] * p for w, p in WEAPON_PRICES.items()
                 if f"ct_weapon_{w}" in df.columns)
    t_val  = sum(df[f"t_weapon_{w}"]  * p for w, p in WEAPON_PRICES.items()
                 if f"t_weapon_{w}"  in df.columns)
    df["ct_weapon_value"] = ct_val
    df["t_weapon_value"]  = t_val
    df["weapon_value_diff"] = ct_val - t_val

    # 手雷总数差
    grenade_cols_ct = [c for c in df.columns if c.startswith("ct_grenade_")]
    grenade_cols_t  = [c for c in df.columns if c.startswith("t_grenade_")]
    df["ct_total_grenades"] = df[grenade_cols_ct].sum(axis=1)
    df["t_total_grenades"]  = df[grenade_cols_t].sum(axis=1)
    df["grenade_diff"]      = df["ct_total_grenades"] - df["t_total_grenades"]

    return df

SURVIVAL_FEATURES = [
    "ct_health","t_health","health_diff",
    "ct_armor","t_armor","armor_diff",
    "ct_players_alive","t_players_alive","players_alive_diff",
    "bomb_planted","time_left",
]

ECONOMIC_FEATURES = [
    "ct_money","t_money","money_diff",
    "ct_helmets","t_helmets","helmets_diff",
    "ct_defuse_kits",
    "ct_weapon_value","t_weapon_value","weapon_value_diff",
    "ct_total_grenades","t_total_grenades","grenade_diff",
]

def split_and_eval(df, feature_cols, label="all"):
    round_ids = make_round_ids(df)
    X = df[feature_cols].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df["round_winner"])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=round_ids))

    sc = StandardScaler()
    X_tr = sc.fit_transform(X[tr_idx])
    X_te = sc.transform(X[te_idx])
    y_tr, y_te = y[tr_idx], y[te_idx]

    model = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    f1  = f1_score(y_te, preds, average="weighted")
    print(f"  [{label}]  n_features={len(feature_cols)}  acc={acc:.4f}  f1={f1:.4f}")
    return acc, f1, model, feature_cols

def main():
    df = pd.read_csv(DATA_PATH)
    df["map"] = LabelEncoder().fit_transform(df["map"])
    df["bomb_planted"] = df["bomb_planted"].astype(int)
    df = engineer_features(df)

    all_features = [c for c in df.columns if c != "round_winner"]

    print("=== Feature Subset Comparison (Random Forest) ===")
    results = {}
    acc, f1, model_all, _ = split_and_eval(df, all_features, "All features")
    results["All features"] = (acc, f1)

    acc, f1, _, _ = split_and_eval(df, SURVIVAL_FEATURES, "Survival only")
    results["Survival only"] = (acc, f1)

    acc, f1, _, _ = split_and_eval(df, ECONOMIC_FEATURES, "Economic only")
    results["Economic only"] = (acc, f1)

    # Feature importance from full model
    print("\n=== Top 20 Feature Importances (All Features model) ===")
    importances = pd.Series(model_all.feature_importances_, index=all_features)
    top20 = importances.nlargest(20)
    for feat, imp in top20.items():
        print(f"  {feat:<35} {imp:.4f}")

    print("\n=== Summary ===")
    for label, (acc, f1) in results.items():
        print(f"  {label:<20} acc={acc:.4f}  f1={f1:.4f}")

if __name__ == "__main__":
    main()
