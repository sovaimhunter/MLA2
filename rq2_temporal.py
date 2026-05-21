"""
RQ2: How does the time point of a round snapshot affect prediction difficulty?
Usage:
  python rq2_temporal.py accuracy    # accuracy by round phase (early/mid/late)
  python rq2_temporal.py importance  # feature importance shift across phases
  python rq2_temporal.py all
"""

import argparse
import pandas as pd
from sklearn.metrics import classification_report
from shared import load_base, split_fit_eval, categorise


def run_accuracy():
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


def run_importance():
    print("\n" + "="*55)
    print("  RQ2: Feature Importance Shift Across Phases")
    print("="*55)

    df = load_base()
    feature_cols = [c for c in df.columns if c != "round_winner"]

    early = df["time_left"] > 120
    mid   = (df["time_left"] >= 60) & (df["time_left"] <= 120)
    late  = df["time_left"] < 60

    results = {}
    for label, mask in [("Early >120s", early), ("Mid 60-120s", mid), ("Late <60s", late)]:
        acc, _, model, cols, _, _ = split_fit_eval(df, feature_cols, mask=mask, n_estimators=300)

        imp    = pd.Series(model.feature_importances_, index=cols)
        imp_df = imp.reset_index()
        imp_df.columns = ["feature", "importance"]
        imp_df["category"] = imp_df["feature"].apply(categorise)
        cat_share = imp_df.groupby("category")["importance"].sum()
        top10     = imp_df.nlargest(10, "importance")

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


RUNNERS = {
    "accuracy":   run_accuracy,
    "importance": run_importance,
}


def main():
    parser = argparse.ArgumentParser(description="RQ2 — CS:GO Round Winner: Temporal Analysis")
    parser.add_argument("task", choices=list(RUNNERS.keys()) + ["all"])
    args = parser.parse_args()
    targets = list(RUNNERS.keys()) if args.task == "all" else [args.task]
    for name in targets:
        RUNNERS[name]()


if __name__ == "__main__":
    main()
