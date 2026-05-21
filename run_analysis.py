"""
CS:GO Round Winner — Analysis & Figures
Usage:
  python run_analysis.py rq2        # accuracy by round phase (early/mid/late)
  python run_analysis.py rq2deep    # feature importance shift across phases
  python run_analysis.py rq3        # feature engineering comparison (selective diffs)
  python run_analysis.py rq3diff    # full diff-ification comparison
  python run_analysis.py figures    # regenerate all report figures
  python run_analysis.py all        # run everything
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report

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


# ── Shared helpers ───────────────────────────────────────────

def make_round_ids(df):
    return (df["time_left"].diff().fillna(0) > 0).cumsum()


def load_base():
    df = pd.read_csv(DATA_PATH)
    df["map"] = LabelEncoder().fit_transform(df["map"])
    df["bomb_planted"] = df["bomb_planted"].astype(int)
    return df


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
    df["ct_weapon_value"]  = ct_val
    df["t_weapon_value"]   = t_val
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


# ── RQ2: Accuracy by round phase ────────────────────────────

def run_rq2():
    print("\n" + "="*55)
    print("  RQ2: Accuracy by Round Phase (Early / Mid / Late)")
    print("="*55)

    df = load_base()
    feature_cols = [c for c in df.columns if c != "round_winner"]

    early = df["time_left"] > 120
    mid   = (df["time_left"] >= 60) & (df["time_left"] <= 120)
    late  = df["time_left"] < 60

    print(f"\nSegment sizes:")
    print(f"  Early (>120s): {early.sum()}")
    print(f"  Mid (60-120s): {mid.sum()}")
    print(f"  Late (<60s):   {late.sum()}")

    results = {}
    for name, mask in [("Early >120s", early), ("Mid 60-120s", mid), ("Late <60s", late)]:
        acc, f1, _, _, y_te, preds = split_fit_eval(df, feature_cols, mask=mask)
        print(f"\n[{name}]  n_test={mask.sum()//5}")
        print(f"  Accuracy: {acc:.4f}  F1: {f1:.4f}")
        print(classification_report(y_te, preds, target_names=["CT", "T"]))
        results[name] = {"acc": acc, "f1": f1, "n": mask.sum()}

    print("\n=== Bomb Planted Rate per Segment ===")
    for name, mask in [("Early", early), ("Mid", mid), ("Late", late)]:
        print(f"  {name}: {df[mask]['bomb_planted'].mean():.3f}")

    print("\n=== Summary ===")
    for name, v in results.items():
        print(f"  {name}: acc={v['acc']:.4f}  f1={v['f1']:.4f}  n={v['n']}")


# ── RQ2 Deep: Feature importance shift across phases ─────────

def run_rq2deep():
    print("\n" + "="*55)
    print("  RQ2 Deep: Feature Importance Shift Across Phases")
    print("="*55)

    df = load_base()
    feature_cols = [c for c in df.columns if c != "round_winner"]

    early = df["time_left"] > 120
    mid   = (df["time_left"] >= 60) & (df["time_left"] <= 120)
    late  = df["time_left"] < 60

    results = {}
    for label, mask in [("Early >120s", early), ("Mid 60-120s", mid), ("Late <60s", late)]:
        acc, _, model, cols, _, _ = split_fit_eval(df, feature_cols, mask=mask, n_estimators=300)

        imp = pd.Series(model.feature_importances_, index=cols)
        imp_df = imp.reset_index()
        imp_df.columns = ["feature", "importance"]
        imp_df["category"] = imp_df["feature"].apply(categorise)
        cat_share = imp_df.groupby("category")["importance"].sum()
        top10 = imp_df.nlargest(10, "importance")

        print(f"\n{'='*50}")
        print(f"Segment: {label}  (n={mask.sum()}, acc={acc:.4f})")
        print("Category importance share:")
        for cat, share in cat_share.sort_values(ascending=False).items():
            print(f"  {cat:<12} {share:.4f}  ({share*100:.1f}%)")
        print("Top 10 features:")
        for _, row in top10.iterrows():
            print(f"  {row['feature']:<35} {row['importance']:.4f}  [{row['category']}]")

        results[label] = {"acc": acc, "cat_share": cat_share}

    print("\n\n=== Economic vs Survival share across segments ===")
    print(f"{'Segment':<15} {'Acc':>6}  {'Economic':>10}  {'Survival':>10}")
    print("-" * 50)
    for label, v in results.items():
        eco = v["cat_share"].get("economic", 0)
        sur = v["cat_share"].get("survival", 0)
        print(f"{label:<15} {v['acc']:>6.4f}  {eco:>10.4f}  {sur:>10.4f}")


# ── RQ3: Selective feature engineering comparison ────────────

SURVIVAL_FEATURES = [
    "ct_health", "t_health", "health_diff",
    "ct_armor", "t_armor", "armor_diff",
    "ct_players_alive", "t_players_alive", "players_alive_diff",
    "bomb_planted", "time_left",
]
ECONOMIC_FEATURES = [
    "ct_money", "t_money", "money_diff",
    "ct_helmets", "t_helmets", "helmets_diff",
    "ct_defuse_kits",
    "ct_weapon_value", "t_weapon_value", "weapon_value_diff",
    "ct_total_grenades", "t_total_grenades", "grenade_diff",
]


def run_rq3():
    print("\n" + "="*55)
    print("  RQ3: Feature Engineering Comparison (Selective Diffs)")
    print("="*55)

    df = load_base()
    df = engineer_features(df)
    all_features = [c for c in df.columns if c != "round_winner"]

    print("\n=== Feature Subset Comparison (Random Forest) ===")
    results = {}

    acc, f1, model_all, _, _, _ = split_fit_eval(df, all_features)
    print(f"  [All features]  n_features={len(all_features)}  acc={acc:.4f}  f1={f1:.4f}")
    results["All features"] = (acc, f1)

    acc, f1, _, _, _, _ = split_fit_eval(df, SURVIVAL_FEATURES)
    print(f"  [Survival only]  n_features={len(SURVIVAL_FEATURES)}  acc={acc:.4f}  f1={f1:.4f}")
    results["Survival only"] = (acc, f1)

    acc, f1, _, _, _, _ = split_fit_eval(df, ECONOMIC_FEATURES)
    print(f"  [Economic only]  n_features={len(ECONOMIC_FEATURES)}  acc={acc:.4f}  f1={f1:.4f}")
    results["Economic only"] = (acc, f1)

    print("\n=== Top 20 Feature Importances (All Features model) ===")
    imp = pd.Series(model_all.feature_importances_, index=all_features).nlargest(20)
    for feat, v in imp.items():
        print(f"  {feat:<35} {v:.4f}")

    print("\n=== Summary ===")
    for label, (acc, f1) in results.items():
        print(f"  {label:<20} acc={acc:.4f}  f1={f1:.4f}")


# ── RQ3 Diff: Full diff-ification comparison ─────────────────

def run_rq3diff():
    print("\n" + "="*55)
    print("  RQ3 Diff: Full Diff-ification Comparison")
    print("="*55)

    df = load_base()
    df_diff, diff_cols = add_all_diffs(df)

    raw_cols       = [c for c in df.columns if c != "round_winner"]
    diff_only_cols = diff_cols + ["time_left", "bomb_planted", "map"]
    all_cols       = [c for c in df_diff.columns if c != "round_winner"]

    print("\n=== Feature Strategy Comparison ===\n")
    results = {}
    for label, cols, data in [
        ("Raw only",        raw_cols,       df),
        ("Diff only",       diff_only_cols, df_diff),
        ("Raw + All diffs", all_cols,       df_diff),
    ]:
        acc, f1, model, _, _, _ = split_fit_eval(data, cols)
        print(f"  [{label}]  n_features={len(cols):>3}  acc={acc:.4f}  f1={f1:.4f}")
        imp = pd.Series(model.feature_importances_, index=cols).nlargest(10)
        print("  Top 10:")
        for feat, v in imp.items():
            print(f"    {feat:<40} {v:.4f}")
        results[label] = (acc, f1)
        print()

    print("=== Summary ===")
    print(f"  {'Strategy':<20} {'Accuracy':>9}  {'F1':>7}")
    print("  " + "-"*42)
    for label, (acc, f1) in results.items():
        print(f"  {label:<20} {acc:>9.4f}  {f1:>7.4f}")


# ── Figures ──────────────────────────────────────────────────

def run_figures():
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    os.makedirs("figures", exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ── hardcoded results for fast re-generation ──
    models     = ["Random Forest", "XGBoost", "Neural Network", "Logistic Regression", "KNN"]
    acc_before = [0.7577, 0.7590, 0.7485, 0.7482, 0.7219]
    acc_after  = [0.7620, 0.7539, 0.7489, 0.7489, 0.7397]

    segments  = ["Early\n(>120s)", "Mid\n(60–120s)", "Late\n(<60s)"]
    seg_acc   = [0.7652, 0.6854, 0.8147]
    eco_share = [0.755, 0.719, 0.506]
    sur_share = [0.220, 0.258, 0.478]

    features    = ["armor_diff", "weapon_value_diff", "health_diff", "helmets_diff",
                   "players_alive_diff", "ct_armor", "ct_weapon_value", "t_armor",
                   "t_weapon_value", "money_diff"]
    importances = [0.1032, 0.0665, 0.0488, 0.0487, 0.0431, 0.0429, 0.0418, 0.0383, 0.0354, 0.0326]
    feat_cats   = ["survival", "economic", "survival", "economic", "survival",
                   "survival", "economic", "survival", "economic", "economic"]
    feat_colors = ["#2196F3" if c == "survival" else "#FF9800" for c in feat_cats]

    # Figure 1: Before vs after tuning
    fig, ax = plt.subplots(figsize=(9, 5))
    x, w = np.arange(len(models)), 0.35
    ax.bar(x - w/2, [v*100 for v in acc_before], w, label="Before tuning",
           color="#90CAF9", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + w/2, [v*100 for v in acc_after], w, label="After tuning",
                   color="#1565C0", edgecolor="white", linewidth=0.8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=9, color="#1565C0", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(68, 80)
    ax.set_title("Figure 1: Model Accuracy Before and After Hyperparameter Tuning", fontsize=12, pad=12)
    ax.legend(frameon=False)
    ax.axhline(75, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(4.6, 75.2, "75%", color="gray", fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/fig1_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig1_model_comparison.png")

    # Figure 2: Temporal analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.plot(segments, [v*100 for v in seg_acc], "o-", color="#1565C0",
             linewidth=2.5, markersize=9, zorder=3)
    for i, (seg, acc) in enumerate(zip(segments, seg_acc)):
        ax1.text(i, acc*100 + 0.5, f"{acc*100:.1f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#1565C0")
    ax1.set_ylim(63, 87)
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Prediction Accuracy by Round Phase", fontsize=11)
    ax1.fill_between(range(3), [v*100 for v in seg_acc], 63, alpha=0.08, color="#1565C0")
    ax1.axhline(76.2, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.text(2.05, 76.5, "Overall\n76.2%", color="gray", fontsize=8)

    bar_w = 0.45
    ax2.bar(range(3), [v*100 for v in eco_share], bar_w, label="Economic", color="#FF9800")
    ax2.bar(range(3), [v*100 for v in sur_share], bar_w,
            bottom=[v*100 for v in eco_share], label="Survival", color="#2196F3")
    other = [100 - (e+s)*100 for e, s in zip(eco_share, sur_share)]
    ax2.bar(range(3), other, bar_w,
            bottom=[(e+s)*100 for e, s in zip(eco_share, sur_share)],
            label="Other", color="#BDBDBD")
    for i, (e, s) in enumerate(zip(eco_share, sur_share)):
        ax2.text(i, e*50, f"{e*100:.0f}%", ha="center", va="center",
                 fontsize=9, color="white", fontweight="bold")
        ax2.text(i, e*100 + s*50, f"{s*100:.0f}%", ha="center", va="center",
                 fontsize=9, color="white", fontweight="bold")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(segments)
    ax2.set_ylabel("Feature Importance Share (%)")
    ax2.set_title("Economic vs Survival Feature Share\nby Round Phase", fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("Figure 2: Temporal Analysis of Prediction Difficulty and Feature Importance",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig2_temporal_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig2_temporal_analysis.png")

    # Figure 3: Feature importances
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importances, color=feat_colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, importances):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)")
    ax.set_title("Figure 3: Top 10 Feature Importances (Random Forest)", fontsize=12, pad=12)
    ax.set_xlim(0, 0.125)
    legend_patches = [
        mpatches.Patch(color="#2196F3", label="Survival"),
        mpatches.Patch(color="#FF9800", label="Economic"),
    ]
    ax.legend(handles=legend_patches, frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig("figures/fig3_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig3_feature_importance.png")

    # Figure 4: Confusion matrix (trains RF on the fly)
    print("Loading data for confusion matrix...")
    df = pd.read_csv(DATA_PATH)
    df["map"] = LabelEncoder().fit_transform(df["map"])
    df["bomb_planted"] = df["bomb_planted"].astype(int)
    round_ids = make_round_ids(df)
    le = LabelEncoder()
    X = df.drop(columns=["round_winner"]).values.astype(float)
    y = le.fit_transform(df["round_winner"])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=round_ids))
    sc = StandardScaler()
    X_tr = sc.fit_transform(X[tr_idx])
    X_te = sc.transform(X[te_idx])
    rf = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=5,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y[tr_idx])
    preds = rf.predict(X_te)
    cm = confusion_matrix(y[te_idx], preds)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Figure 4: Confusion Matrix — Random Forest (Test Set)", fontsize=11, pad=12)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j, i + 0.35, f"({pct:.1f}%)", ha="center", va="center",
                    fontsize=9, color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig("figures/fig4_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig4_confusion_matrix.png")

    print("\nAll figures saved to figures/")


# ── Entry point ──────────────────────────────────────────────

RUNNERS = {
    "rq2":     run_rq2,
    "rq2deep": run_rq2deep,
    "rq3":     run_rq3,
    "rq3diff": run_rq3diff,
    "figures": run_figures,
}


def main():
    parser = argparse.ArgumentParser(description="CS:GO Round Winner — Analysis & Figures")
    parser.add_argument("task", choices=list(RUNNERS.keys()) + ["all"],
                        help="Analysis task to run, or 'all' to run everything")
    args = parser.parse_args()

    targets = list(RUNNERS.keys()) if args.task == "all" else [args.task]
    for name in targets:
        RUNNERS[name]()


if __name__ == "__main__":
    main()
