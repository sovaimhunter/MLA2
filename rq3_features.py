"""
RQ3: How do economic, survival, and relative-advantage features differ in predictive contribution?
Usage:
  python rq3_features.py selective   # survival vs economic vs all features
  python rq3_features.py diff        # raw vs diff-only vs raw+diff strategies
  python rq3_features.py all
"""

import argparse
import pandas as pd
from shared import load_base, engineer_features, add_all_diffs, split_fit_eval

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


def run_selective():
    print("\n" + "="*55)
    print("  RQ3: Feature Engineering Comparison (Selective Diffs)")
    print("="*55)

    df = load_base()
    df = engineer_features(df)
    all_features = [c for c in df.columns if c != "round_winner"]

    print("\n=== Feature Subset Comparison (Random Forest) ===")
    results = {}

    acc, f1, model_all, _, _, _ = split_fit_eval(df, all_features)
    print(f"  [All features]   n_features={len(all_features)}  acc={acc:.4f}  f1={f1:.4f}")
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


def run_diff():
    print("\n" + "="*55)
    print("  RQ3: Full Diff-ification Comparison")
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


RUNNERS = {
    "selective": run_selective,
    "diff":      run_diff,
}


def main():
    parser = argparse.ArgumentParser(description="RQ3 — CS:GO Round Winner: Feature Engineering")
    parser.add_argument("task", choices=list(RUNNERS.keys()) + ["all"])
    args = parser.parse_args()
    targets = list(RUNNERS.keys()) if args.task == "all" else [args.task]
    for name in targets:
        RUNNERS[name]()


if __name__ == "__main__":
    main()
