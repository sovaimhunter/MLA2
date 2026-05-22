"""
CS:GO Round Winner — Report Figures
Usage:
  python run_analysis.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from shared import load_base, make_round_ids, load_data


def run_figures():
    os.makedirs("figures", exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ── results from rq1_models.py all / rq2_temporal.py all ──
    models     = ["Random Forest", "XGBoost", "Neural Network", "Logistic Regression"]
    acc_before = [0.7559, 0.7530, 0.7478, 0.7482]
    acc_after  = [0.7602, 0.7543, 0.7496, 0.7489]

    segments  = ["Early\n(>120s)", "Mid\n(60–120s)", "Late\n(<60s)"]
    seg_acc   = [0.7643, 0.6852, 0.8147]
    eco_share = [0.749, 0.717, 0.492]
    sur_share = [0.228, 0.261, 0.493]

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
    ax.legend(frameon=False)
    ax.axhline(75, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(3.6, 75.2, "75%", color="gray", fontsize=9)
    ax.axhline(76.02, color="#1565C0", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.text(3.6, 76.2, "Best\n76.0%", color="#1565C0", fontsize=8)
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
    ax1.fill_between(range(3), [v*100 for v in seg_acc], 63, alpha=0.08, color="#1565C0")
    ax1.axhline(76.02, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.text(2.05, 76.3, "Overall\n76.0%", color="gray", fontsize=8)

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
    ax2.set_ylim(0, 110)
    ax2.legend(frameon=False, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/fig2_temporal_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig2_temporal_analysis.png")

    # Figure 3: Feature importances
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(features))
    bars  = ax.barh(y_pos, importances, color=feat_colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, importances):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)")
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
    df = load_base()
    round_ids = make_round_ids(df)
    le = LabelEncoder()
    X  = df.drop(columns=["round_winner"]).values.astype(float)
    y  = le.fit_transform(df["round_winner"])

    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=round_ids))
    sc = StandardScaler()
    X_tr = sc.fit_transform(X[tr_idx])
    X_te = sc.transform(X[te_idx])

    rf = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=2,
                                min_samples_leaf=1, max_features=0.5,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y[tr_idx])
    preds = rf.predict(X_te)
    cm    = confusion_matrix(y[te_idx], preds)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
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

    # Figure 3 (error analysis): hardcoded from rq3_errors.py run
    situations   = ["Close\nRounds", "One-Sided\nRounds", "Comeback\nRounds"]
    sit_acc      = [73.3, 95.7, 27.5]
    sit_n        = [16655, 6279, 1523]
    sit_colors   = ["#64B5F6", "#1565C0", "#E53935"]
    conf_low     = [31, 47, 29]
    conf_mid     = [38, 12, 42]
    conf_high    = [31, 41, 29]
    overall_acc  = 76.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar(situations, sit_acc, color=sit_colors, edgecolor="white", linewidth=0.8, width=0.5)
    for bar, acc, n, col in zip(bars, sit_acc, sit_n, sit_colors):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5,
                 f"{acc:.1f}%", ha="center", va="top", fontsize=12,
                 fontweight="bold", color="white" if acc > 40 else col)
        ax1.text(bar.get_x() + bar.get_width()/2, 3,
                 f"n={n:,}", ha="center", va="bottom", fontsize=9, color="white")
    ax1.axhline(overall_acc, color="gray", linestyle="--", linewidth=1.2)
    ax1.text(-0.45, overall_acc + 1.5, f"Overall acc ({overall_acc}%)", color="gray", fontsize=9)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 110)

    x = np.arange(len(situations))
    w = 0.5
    b1 = ax2.bar(x, conf_low,  w, label="Low conf (<0.4)",    color="#F48FB1")
    b2 = ax2.bar(x, conf_mid,  w, bottom=conf_low,            label="Uncertain (0.4–0.6)", color="#FFF176")
    b3 = ax2.bar(x, conf_high, w, bottom=[l+m for l, m in zip(conf_low, conf_mid)],
                 label="High conf (>0.6)", color="#A5D6A7")
    for i, (lo, mi, hi) in enumerate(zip(conf_low, conf_mid, conf_high)):
        ax2.text(i, lo/2,        f"{lo}%", ha="center", va="center", fontsize=9, fontweight="bold")
        ax2.text(i, lo + mi/2,   f"{mi}%", ha="center", va="center", fontsize=9, fontweight="bold")
        ax2.text(i, lo + mi + hi/2, f"{hi}%", ha="center", va="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(situations)
    ax2.set_ylabel("Share of Snapshots (%)")
    ax2.set_ylim(0, 115)
    ax2.legend(frameon=False, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/fig3_error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig3_error_analysis.png")

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    run_figures()
