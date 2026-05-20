# CS:GO Round Winner Prediction — Research Findings

Dataset: [CS:GO Round Winner Classification](https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification)  
Rows: 122,410 | Features: 96 | Task: Binary classification (CT vs T)  
Train/Test split: **Group-based** (by round ID, 80/20) — prevents same-round snapshots leaking across splits.

---

## RQ1: Which ML model best predicts round winner?

### Setup
Five classifiers compared with hyperparameter tuning (GridSearchCV / RandomizedSearchCV, 3-fold CV):

| Model | Bias–Variance | Tuned Hyperparameters |
|---|---|---|
| Logistic Regression | High bias / Low variance | C, solver |
| Random Forest | Low bias / Low variance | n_estimators, max_depth, min_samples_split |
| XGBoost | Low bias / Low variance | max_depth, learning_rate, n_estimators, subsample, colsample_bytree |
| Neural Network | Medium | learning_rate, dropout, hidden_dim |
| KNN | Medium bias / High variance | n_neighbors (k), metric |

### Results

| Model | Before Tuning | After Tuning | Best Hyperparameters |
|---|---|---|---|
| **Random Forest** | 75.77% | **76.20%** | n_estimators=500, max_depth=20, min_samples_split=5 |
| XGBoost | 75.90% | 75.39% | lr=0.01, max_depth=5, n_estimators=300 |
| Neural Network | 74.85% | 74.89% | lr=0.001, dropout=0.4, hidden_dim=128 |
| Logistic Regression | 74.82% | 74.89% | C=0.01, solver=lbfgs |
| KNN | 72.19% | 73.97% | k=21, metric=manhattan |

### Key Findings

- **Random Forest wins** after tuning, demonstrating that ensemble methods with sufficient trees (500) generalise well on this tabular, feature-rich dataset.
- **XGBoost underperformed after tuning** — its default settings happened to be near-optimal; the RandomizedSearch (15 iterations) was insufficient to cover the full search space and found a slower-learning configuration (lr=0.01) that did not benefit from only 300 trees.
- **LR and NN are tied**, both at ~74.9%. LR's strong regularisation (C=0.01) suggests the feature space benefits from shrinking weak predictors. NN adds no advantage here — the dataset is not large enough and lacks spatial/sequential structure.
- **KNN improved most from tuning (+1.78%)** because manhattan distance is more robust than euclidean in high-dimensional sparse feature spaces (most weapon-count features are 0).
- **Data leakage note**: a naive random split inflated all scores by allowing snapshots from the same round to appear in both train and test. Fixing to group-based split reduced scores by ~1–2% and reflects true generalisation.

---

## RQ2: How does snapshot timing affect prediction difficulty?

### Setup
Data split into three temporal segments, each evaluated independently using the best model (Random Forest with optimal hyperparameters):

| Segment | time_left range | Rows | Bomb Planted Rate |
|---|---|---|---|
| Early | > 120s | 33,715 | 0.0% |
| Mid | 60–120s | 52,064 | 0.0% |
| Late | < 60s | 36,631 | 37.4% |

### Results

| Segment | Accuracy | F1 (weighted) |
|---|---|---|
| **Late (<60s)** | **81.35%** | **0.813** |
| Early (>120s) | 76.52% | 0.765 |
| Mid (60–120s) | **68.54%** | **0.684** |

### Deep Analysis: Feature Importance by Segment

To verify *why* early-round accuracy is unexpectedly high, feature importance was extracted per segment:

| Segment | Accuracy | Economic share | Survival share |
|---|---|---|---|
| Early (>120s) | 76.52% | **75.5%** | 22.0% |
| Mid (60–120s) | 68.43% | 71.9% | 25.8% |
| Late (<60s) | 81.47% | 50.6% | **47.8%** |

Early-segment top features: `t_armor`, `ct_armor`, `t_money`, `ct_money`, `t_helmets`, `ct_defuse_kits`  
Late-segment top features: `ct_armor`, `t_armor`, `t_health`, `ct_health`, `t_players_alive`, `bomb_planted`

### Key Findings

- **Late-round snapshots are the easiest to predict (81.4%)** — survival features surge to 47.8% importance as player counts and health diverge sharply. With 37.4% of late snapshots having the bomb planted, the tactical state is highly constrained.
- **Mid-round is the hardest (68.5%)**, 12.8pp below late-round. Both teams are still at full strength with varied equipment; individual skill and positioning dominate — factors absent from snapshot features.
- **Early-round hypothesis confirmed**: economic features account for 75.5% of importance in the early segment — the highest across all three. This validates the hypothesis that **economic inertia from previous rounds** (pistol round winner gains rifle advantage in rounds 2–3) is the dominant signal when no combat has occurred. The model is effectively reading the buy-phase outcome.
- **Feature importance shifts with time**: the round transitions from an *economic prediction problem* (early) → an *unpredictable tactical problem* (mid) → a *survival state prediction problem* (late). This three-phase structure is a key insight for future work.

---

## RQ3: Which feature category matters most — economic or survival?

### Feature Engineering
Beyond the original 96 raw features, difference and aggregate features were constructed. All ct_/t_ paired columns were automatically differenced to test whether full diff-ification improves performance:

| Feature | Formula | Category |
|---|---|---|
| health_diff | ct_health − t_health | Survival |
| armor_diff | ct_armor − t_armor | Survival |
| players_alive_diff | ct_players_alive − t_players_alive | Survival |
| money_diff | ct_money − t_money | Economic |
| helmets_diff | ct_helmets − t_helmets | Economic |
| weapon_value_diff | Σ(ct weapons × price) − Σ(t weapons × price) | Economic |
| ct/t_weapon_value | Total weapon loadout value per side | Economic |
| grenade_diff | ct grenades − t grenades | Economic |

### Feature Strategy Comparison (Random Forest)

| Feature Set | # Features | Accuracy | F1 |
|---|---|---|---|
| Raw only (original 96) | 96 | 76.20% | 0.762 |
| Diff only (all ct−t diffs + weapon value) | 52 | 75.59% | 0.756 |
| Raw + All diffs | 145 | 75.73% | 0.757 |
| Survival subset only | 11 | 74.07% | 0.740 |
| Economic subset only | 13 | 73.55% | 0.735 |

### Top 20 Feature Importances

| Rank | Feature | Importance | Category |
|---|---|---|---|
| 1 | armor_diff | 0.1032 | Survival |
| 2 | weapon_value_diff | 0.0665 | Economic |
| 3 | health_diff | 0.0488 | Survival |
| 4 | helmets_diff | 0.0487 | Economic |
| 5 | players_alive_diff | 0.0431 | Survival |
| 6 | ct_armor | 0.0429 | Survival |
| 7 | ct_weapon_value | 0.0418 | Economic |
| 8 | t_armor | 0.0383 | Survival |
| 9 | t_weapon_value | 0.0354 | Economic |
| 10 | money_diff | 0.0326 | Economic |
| 11 | t_money | 0.0303 | Economic |
| 12 | ct_money | 0.0296 | Economic |
| 13 | grenade_diff | 0.0274 | Economic |
| 14 | time_left | 0.0222 | Context |
| 15 | t_helmets | 0.0215 | Economic |
| 16 | t_score | 0.0175 | Context |
| 17 | ct_total_grenades | 0.0173 | Economic |
| 18 | ct_score | 0.0171 | Context |
| 19 | t_total_grenades | 0.0161 | Economic |
| 20 | t_health | 0.0158 | Survival |

### Key Findings

- **Survival features edge out economic features** (74.1% vs 73.6%) despite using fewer raw values. In-round state (who is alive, how much health/armor) is a more direct signal of who will win the current round.
- **armor_diff is the single most important feature (10.3%)** — far ahead of everything else. Armor is a direct combat advantage: an armored player takes significantly less damage from rifle shots, and its absence is difficult to compensate for with money alone.
- **weapon_value_diff ranks 2nd (6.65%)**, confirming that economic investment translates directly to firepower advantage.
- **Engineered difference features dominate the top 5** — all five top features are differences (diff), not raw values. This confirms that **relative advantage** between teams matters more than absolute values. A team with $4000 cash is irrelevant; what matters is whether the opponent has $1000 or $8000.
- **All-feature model (75.74%) outperforms both subsets**, suggesting survival and economic signals are complementary rather than redundant. The gap over survival-only (~1.7pp) shows economic context still adds meaningful signal.
- **Individual weapon columns (e.g., ct_weapon_ak47) rank low** — what matters is total weapon value, not which specific weapon. This validates the feature engineering choice to aggregate by price.
- **Full diff-ification does not help**: replacing raw features with all-diff features drops accuracy from 76.2% → 75.6%. Adding diffs on top of raw features (145 features) also underperforms raw-only. This is counter-intuitive given that individual diff features rank highly — the reason is that raw features still carry side-specific information (e.g., CT tends to have higher armor than T on average) that diffs discard. **Selective feature engineering** (computing diffs only for the most important pairs) outperforms bulk transformation.

---

## Overall Summary

| Research Question | Answer |
|---|---|
| RQ1 Best model | **Random Forest** (76.2%) after tuning, narrowly over XGBoost |
| RQ2 Timing effect | Mid-round hardest (68.5%), Late-round easiest (81.4%) — 12.8pp gap |
| RQ3 Feature type | Survival slightly > Economic, but **difference features** are universally most important |

**Performance ceiling**: All models cluster around 74–76% on full data. This reflects the inherent randomness of mid-game CS:GO — many outcomes are decided by individual skill, game sense, and positioning that no snapshot-level feature can capture.

**Extended findings**:
- The round has three structurally distinct phases with different dominant feature types (economic → chaotic → survival), confirmed quantitatively via per-segment feature importance
- Full diff-ification of features is counterproductive — raw features contain side-asymmetric information that diffs lose; only selective diffs (armor, weapon value, health, players) add value
- Future work: incorporating player position data or time-series of snapshots (rather than single snapshots) would directly address the mid-round unpredictability gap
