"""
RQ4: Are model errors concentrated in uncertain or comeback-like situations?

Classifies test-set snapshots into:
  - close      : differences in money/health/alive all near zero
  - one_sided  : one team clearly ahead AND wins (expected outcome)
  - comeback   : clearly disadvantaged team wins (upset)

Then compares model error rate across the three types.

Usage:
  python rq4_errors.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from shared import make_round_ids, WEAPON_PRICES

DATA_PATH = "csgo_round_snapshots.csv"
TEST_SIZE  = 0.2
RANDOM_STATE = 42

# Thresholds for "close" classification
MONEY_CLOSE_THR  = 1000   # |ct_money - t_money| <= this
HEALTH_CLOSE_THR = 20     # |ct_health - t_health| <= this
# alive_diff must be 0 for "close"

# Advantage score >= this to be "one-sided" or "comeback"
ADV_THR = 2   # out of 3 dimensions (alive, money, health)


def classify_situation(alive_diff, money_diff, health_diff, ct_wins):
    """
    Returns 'close', 'one_sided', or 'comeback'.
    ct_wins: True if CT won the round.
    """
    adv = int(np.sign(alive_diff)) + int(np.sign(money_diff)) + int(np.sign(health_diff))
    # adv in [-3, 3]: positive = CT favoured, negative = T favoured

    if abs(adv) < ADV_THR:
        return "close"

    ct_favoured = adv > 0
    if ct_favoured == ct_wins:
        return "one_sided"
    else:
        return "comeback"


def run():
    print("\n" + "="*60)
    print("  RQ4 — Model Error Analysis by Round Situation")
    print("="*60)

    # ── Load raw data (for diff computation) ──────────────────
    df = pd.read_csv(DATA_PATH)
    df["map"] = LabelEncoder().fit_transform(df["map"])
    df["bomb_planted"] = df["bomb_planted"].astype(int)

    round_ids = make_round_ids(df)
    print(f"Total snapshots: {len(df)}  |  Total rounds: {round_ids.nunique()}")

    # Diff features (on raw, unscaled data)
    df["money_diff"]  = df["ct_money"]         - df["t_money"]
    df["health_diff"] = df["ct_health"]         - df["t_health"]
    df["alive_diff"]  = df["ct_players_alive"]  - df["t_players_alive"]
    df["score_diff"]  = df["ct_score"]          - df["t_score"]

    le = LabelEncoder()
    y_all = le.fit_transform(df["round_winner"])   # CT=0, T=1 (alphabetical)
    ct_label = list(le.classes_).index("CT")

    feature_cols = [c for c in df.columns
                    if c not in ["round_winner", "money_diff", "health_diff",
                                 "alive_diff", "score_diff"]]
    X_all = df[feature_cols].values.astype(np.float32)

    # ── Group-aware train/test split ───────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(X_all, y_all, groups=round_ids))

    sc = StandardScaler()
    X_train = sc.fit_transform(X_all[tr_idx])
    X_test  = sc.transform(X_all[te_idx])
    y_train, y_test = y_all[tr_idx], y_all[te_idx]

    # ── Train Random Forest ────────────────────────────────────
    print("\nTraining Random Forest (n_estimators=500, max_depth=20)...")
    model = RandomForestClassifier(
        n_estimators=500, max_depth=20,
        min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds  = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, ct_label]   # P(CT wins)

    overall_acc = accuracy_score(y_test, preds)
    overall_f1  = f1_score(y_test, preds, average="weighted")
    print(f"Overall  acc={overall_acc:.4f}  f1={overall_f1:.4f}")

    # ── Classify test snapshots into situation types ───────────
    te_df = df.iloc[te_idx].copy()
    te_df["pred"]      = preds
    te_df["label"]     = y_test
    te_df["proba_ct"]  = probas
    te_df["correct"]   = (preds == y_test).astype(int)
    te_df["ct_wins"]   = (y_test == ct_label)

    te_df["situation"] = te_df.apply(
        lambda r: classify_situation(
            r["alive_diff"], r["money_diff"], r["health_diff"], r["ct_wins"]
        ), axis=1
    )

    # ── Results ────────────────────────────────────────────────
    order = ["close", "one_sided", "comeback"]

    print("\n=== Situation Distribution ===")
    total_te = len(te_df)
    for sit in order:
        n = (te_df["situation"] == sit).sum()
        print(f"  {sit:<12}: {n:>6}  ({n/total_te*100:.1f}%)")

    print("\n=== Error Rate by Situation ===")
    print(f"  {'Situation':<12} {'N':>6} {'Acc':>8} {'Error%':>8} {'Avg P(CT)':>10} {'CT_winrate':>12}")
    print("  " + "-"*58)
    for sit in order:
        sub = te_df[te_df["situation"] == sit]
        if len(sub) == 0:
            continue
        acc      = sub["correct"].mean()
        err_rate = 1 - acc
        avg_p    = sub["proba_ct"].mean()
        ct_wr    = sub["ct_wins"].mean()
        print(f"  {sit:<12} {len(sub):>6} {acc:>8.4f} {err_rate*100:>7.1f}% {avg_p:>10.4f} {ct_wr:>12.4f}")

    # ── Confidence distribution per situation ──────────────────
    print("\n=== Model Confidence (P(CT wins)) Distribution ===")
    print(f"  {'Situation':<12} {'Mean':>8} {'Std':>8} {'<0.4':>8} {'0.4-0.6':>10} {'>0.6':>8}")
    print("  " + "-"*58)
    for sit in order:
        sub = te_df[te_df["situation"] == sit]
        if len(sub) == 0:
            continue
        p = sub["proba_ct"]
        low  = (p < 0.4).mean()
        mid  = ((p >= 0.4) & (p <= 0.6)).mean()
        high = (p > 0.6).mean()
        print(f"  {sit:<12} {p.mean():>8.4f} {p.std():>8.4f} {low*100:>7.1f}% {mid*100:>9.1f}% {high*100:>7.1f}%")

    # ── Comeback breakdown ─────────────────────────────────────
    print("\n=== Comeback Rounds: Which Dimension Was the Upset? ===")
    cb = te_df[te_df["situation"] == "comeback"].copy()
    if len(cb) > 0:
        cb["alive_upset"]  = cb["alive_diff"].apply(
            lambda x: "CT_behind" if x < 0 else ("T_behind" if x > 0 else "equal"))
        cb["money_upset"]  = cb["money_diff"].apply(
            lambda x: "CT_behind" if x < 0 else ("T_behind" if x > 0 else "equal"))
        cb["health_upset"] = cb["health_diff"].apply(
            lambda x: "CT_behind" if x < 0 else ("T_behind" if x > 0 else "equal"))

        print(f"  Total comeback snapshots: {len(cb)}")
        print(f"  Model error rate in comeback: {1 - cb['correct'].mean():.3f}")
        print(f"  CT wins comeback: {cb['ct_wins'].mean():.3f}  T wins comeback: {1-cb['ct_wins'].mean():.3f}")
        print(f"\n  Comeback dimension breakdown (who was behind but won):")
        for dim, col in [("alive_diff", "alive_upset"),
                          ("money_diff", "money_upset"),
                          ("health_diff", "health_upset")]:
            vc = cb[col].value_counts()
            print(f"    {dim:<15}: {dict(vc)}")


if __name__ == "__main__":
    run()
