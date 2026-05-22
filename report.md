# Predicting CS:GO Round Winners Using Machine Learning: A Snapshot-Based Approach

**Group Members:** [Name 1], [Name 2], [Name 3]

---

## 1. Introduction

Counter-Strike: Global Offensive (CS:GO) is one of the world's most played competitive first-person shooters, with a large professional esports scene and millions of active players. Each match consists of rounds in which two teams—Counter-Terrorists (CT) and Terrorists (T)—compete to either eliminate the opposing team or complete map-specific objectives. The outcome of each round depends on a complex mix of economic decisions (what equipment to buy), survival state (health, armor, player count), and real-time tactical execution.

Predicting round outcomes from mid-round snapshots is both practically interesting and technically challenging. From a sports analytics perspective, such predictions could power real-time win probability displays, assist coaching staff in identifying critical moments, or support broadcast commentary. From a machine learning perspective, the dataset offers a rich tabular feature space with temporal structure, making it well-suited for exploring how different model families and feature engineering strategies behave on structured game data.

This report addresses three research questions:

- **RQ1:** How do different categories of machine learning models compare in predicting CS:GO round winners, and what does their relative performance reveal about the structure of the data?
- **RQ2:** How does the temporal position of a round snapshot affect prediction difficulty, and which feature types become more or less important as the round progresses?
- **RQ3:** Are model errors concentrated in uncertain or comeback-like situations, and what does the error distribution reveal about the limits of snapshot-based prediction?

These questions are designed not just to find the best model, but to understand *why* certain approaches work—linking model behavior to the domain properties of competitive CS:GO.

---

## 2. Literature Review

The dataset used in this project is a Kaggle dataset of CS:GO round snapshots compiled by Lillelund (2020), containing per-snapshot game state information drawn from professional and semi-professional matches. Each snapshot captures the full equipment and health state of both teams at a point in time within a round.

Esports outcome prediction has received growing academic interest. Semenov et al. (2016) applied multiple classifiers—including logistic regression, naive Bayes, and gradient boosting—to predict match outcomes in Dota 2 from pre-game draft information alone, achieving up to 71.4% accuracy. Their core finding that ensemble methods outperform linear models on high-dimensional, sparse feature spaces is consistent with our observations. A key difference is that they predict at the match level before the game starts, while we predict at the round level from mid-game snapshots, making our task more granular and more dependent on real-time state.

Xenopoulos et al. (2020) analyzed CS:GO specifically, developing action-value metrics to quantify the impact of individual player actions on round outcomes. Their work established that CS:GO game state is highly structured and that survival-related events—kills, bomb plants, and health changes—carry disproportionate predictive weight. Their finding that individual skill constitutes irreducible uncertainty motivates our RQ3 error analysis, which asks whether model failures cluster in structurally disadvantaged or comeback-like situations.

The use of random forests for tabular sports prediction is well-established. Breiman (2001) introduced the algorithm and demonstrated its robustness to noise and correlated features through bootstrap aggregation and random feature subsampling—properties particularly relevant for our 96-feature dataset, which contains many correlated weapon columns.

---

## 3. Method

### 3.1 Dataset

The dataset contains **122,410 snapshots** from CS:GO matches, each representing the complete game state at one point in time during a round. There are **96 original features** covering both teams':

- Health and armor values
- Number of players alive
- Money and helmet counts
- Defuse kit availability (CT only)
- Individual weapon counts (34 weapon types per side)
- Grenade counts (6 grenade types per side)
- Round score, time remaining, map, and bomb-planted status

The target variable is `round_winner` (CT or T), a **binary classification** task. The class distribution is approximately balanced—CT wins: 49.0%, T wins: 51.0%—so no oversampling was required.

The feature space is **predominantly sparse**: for most snapshots, the majority of weapon columns are zero since teams carry only a handful of weapons at once. It also contains **correlated features**, such as `ct_health` and `t_health`, which are individually informative but negatively correlated with each other from the perspective of CT winning. These properties—sparsity and correlation—motivated our choice of ensemble methods over purely linear ones.

### 3.2 Validation Strategy

A critical design choice was the train/test split. A naive random split would allow snapshots from the *same round* to appear in both training and test sets. Since consecutive snapshots within a round share nearly identical game states, this would constitute data leakage—artificially inflating accuracy by 1–2% without reflecting true generalization to unseen rounds.

To prevent this, we used **GroupShuffleSplit** with round IDs as groups (derived from resets in `time_left`). This guarantees that all snapshots belonging to a given round appear exclusively in either training or test. The split was **80/20**, yielding:

- Training set: ~97,953 snapshots
- Test set: ~24,457 snapshots (spanning ~4,707 held-out rounds)
- Total unique rounds: **23,537**

All four models were evaluated on the same test split to ensure fair comparison.

For RQ2, the data was further divided into three temporal segments based on `time_left`:

| Segment | time_left | n snapshots | Bomb Planted Rate |
|---|---|---|---|
| Early | > 120s | 33,715 | 0.0% |
| Mid | 60–120s | 52,064 | 0.0% |
| Late | < 60s | 36,631 | 37.4% |

Each segment was split independently using the same group-based strategy.

#### Feature Category Definitions

To interpret feature importance by phase (RQ2), each feature is assigned to one of three categories based on keyword matching against its column name:

| Category | Keywords matched | Example features |
|---|---|---|
| **Economic** | money, helmet, defuse, weapon, grenade, score | ct_money, t_helmets, ct_defuse_kits, ct_weapon_ak47, t_grenade_flashbang, ct_score |
| **Survival** | health, armor, players_alive, bomb_planted, time_left | ct_health, t_armor, ct_players_alive, bomb_planted, time_left |
| **Other** | (none of the above) | map |

Category importance is computed as the sum of Random Forest feature importances for all features in that category, then normalised to sum to 100% within each segment.

### 3.3 Round Situation Classification

To investigate where model errors concentrate, we classify each test-set snapshot into one of three mutually exclusive situation types based on three difference variables:

| Variable | Formula | Meaning |
|---|---|---|
| `alive_diff` | ct_players_alive − t_players_alive | Player count advantage |
| `money_diff` | ct_money − t_money | Economic advantage |
| `health_diff` | ct_health − t_health | Combat health advantage |

An **advantage score** is computed as the sum of the signs of these three differences, yielding an integer in [−3, +3], where positive values favour CT and negative values favour T. Snapshots are then classified as follows:

| Situation | Definition |
|---|---|
| **close** | \|advantage score\| < 2 — neither team has a clear edge across dimensions |
| **one_sided** | \|advantage score\| ≥ 2 AND the favoured team wins — expected outcome |
| **comeback** | \|advantage score\| ≥ 2 AND the underdog wins — upset outcome |

This classification separates structurally ambiguous rounds (close) from dominant rounds (one_sided) and upset rounds (comeback), allowing us to test whether model errors are driven by genuine uncertainty or by situations that appear certain but resolve unexpectedly.

### 3.4 Models

We selected four models spanning the bias-variance spectrum:

| Model | Bias / Variance | Role |
|---|---|---|
| Logistic Regression | High bias / Low variance | Linear baseline; tests approximate linear separability |
| Neural Network (MLP) | Medium | Tests whether additional non-linearity helps |
| XGBoost | Low bias / Low variance | Gradient boosting; efficient on sparse features |
| Random Forest | Low bias / Low variance | Ensemble with random subsampling; robust to correlated features |

**Logistic Regression** serves as the interpretable linear baseline. If LR performs comparably to complex models, it indicates the decision surface is approximately linear. Hyperparameters tuned: regularization strength `C` and solver.

**Random Forest** aggregates many decorrelated trees via bootstrap sampling and random feature subsampling. The `max_features` parameter controls how many features each split considers—lower values reduce correlation between trees. Hyperparameters tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.

**XGBoost** builds trees sequentially, each correcting residual errors of the previous. The `hist` tree method is efficient for large datasets. Hyperparameters tuned: `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`.

**Neural Network** is a 3-layer MLP with batch normalization and dropout: Linear → BatchNorm → ReLU → Dropout → Linear(hidden/2) → BatchNorm → ReLU → Dropout → Linear(64) → Linear(1). Hyperparameters tuned: `learning_rate`, `dropout`, `hidden_dim`.

Tuning used **RandomizedSearchCV** (30 iterations, 3-fold CV) for RF and XGBoost, **GridSearchCV** (8 combinations, 3-fold CV) for LR, and **grid search over 12 combinations** with a 15%-held-out validation set for NN.

### 3.5 Evaluation

We report **accuracy**, **weighted F1-score**, and per-class **precision and recall** for all models. The confusion matrix for the best model provides insight into CT vs. T prediction asymmetry. Accuracy is appropriate given the near-balanced class distribution; weighted F1 accounts for the slight class imbalance (~51% T vs. ~49% CT). For RQ2, per-segment accuracy captures prediction difficulty over time. For RQ3, **error rate** and **model confidence** (predict_proba) are compared across the three situation types; the confidence distribution reveals whether errors stem from model uncertainty or from structurally misleading game states.

---

## 4. Results

### 4.1 RQ1: Model Comparison

Table 1 presents all four models before and after hyperparameter tuning. Figure 1 visualizes the before/after comparison.

**Table 1: Model Accuracy and F1 Before and After Hyperparameter Tuning**

| Model | Acc (Default) | Acc (Tuned) | F1 (Tuned) | Best CV Acc |
|---|---|---|---|---|
| **Random Forest** | 0.7559 | **0.7602** | **0.7597** | — |
| XGBoost | 0.7530 | 0.7543 | 0.7541 | 0.7505 |
| Neural Network | 0.7478 | 0.7496 | 0.7496 | 0.7510 (val) |
| Logistic Regression | 0.7482 | 0.7489 | 0.7490 | 0.7472 |

**Best hyperparameters found:**
- **RF:** n_estimators=500, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features=0.5
- **XGBoost:** learning_rate=0.05, max_depth=5, n_estimators=300, subsample=0.7, colsample_bytree=0.8, min_child_weight=1, reg_alpha=0
- **LR:** C=0.01, solver=lbfgs
- **NN:** learning_rate=0.001, dropout=0.2, hidden_dim=128

**Table 2: Classification Report — Random Forest (Tuned)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CT | 0.72 | 0.82 | 0.77 | 11,799 |
| T  | 0.81 | 0.71 | 0.75 | 12,658 |
| **Weighted avg** | **0.77** | **0.76** | **0.76** | **24,457** |

The confusion matrix (Figure 4) reveals a systematic asymmetry: CT recall (0.82) is notably higher than T recall (0.71).

### 4.2 RQ2: Temporal Analysis

**Table 3: Prediction Accuracy by Round Phase**

| Segment | n | Accuracy | F1 (weighted) | Bomb Planted |
|---|---|---|---|---|
| Early (>120s) | 33,715 | 0.7643 | 0.7644 | 0% |
| Mid (60–120s) | 52,064 | **0.6852** | **0.6833** | 0% |
| **Late (<60s)** | 36,631 | **0.8147** | **0.8143** | 37.4% |

**Table 4: Feature Category Importance by Phase**

| Segment | Accuracy | Economic | Survival | Other |
|---|---|---|---|---|
| Early (>120s) | 76.6% | **74.9%** | 22.8% | 2.3% |
| Mid (60–120s) | 68.5% | **71.7%** | 26.1% | 2.2% |
| Late (<60s) | 81.4% | 49.2% | **49.3%** | 1.4% |

Figure 2 shows both the accuracy trend across phases and the stacked feature importance breakdown.

### 4.3 RQ3: Error Analysis by Round Situation

**Table 5: Model Performance by Round Situation Type**

| Situation | N (test) | Share | Accuracy | Error Rate | Avg P(CT wins) |
|---|---|---|---|---|---|
| close | 16,655 | 68.1% | 0.7335 | 26.7% | 0.502 |
| one_sided | 6,279 | 25.7% | **0.9568** | **4.3%** | 0.460 |
| **comeback** | 1,523 | 6.2% | **0.2751** | **72.5%** | 0.500 |
| Overall | 24,457 | — | 0.7623 | 23.8% | — |

**Table 6: Model Confidence Distribution by Situation**

| Situation | Low conf (<0.4) | Uncertain (0.4–0.6) | High conf (>0.6) |
|---|---|---|---|
| close | 30.9% | 37.8% | 31.3% |
| one_sided | 47.1% | 12.1% | 40.8% |
| comeback | 29.3% | **41.8%** | 28.9% |

Figure 3 shows accuracy by situation and the confidence distribution across all three types.

---

## 5. Discussion

### 5.1 RQ1: Why Do All Models Cluster Around 75%?

The most striking finding of RQ1 is how narrow the performance gap is across four fundamentally different model families: all fall within a 1.1pp range (74.9%–76.0%) after tuning. This is not a modeling failure—it is a **data ceiling**. CS:GO rounds are decided by factors completely absent from snapshot data: individual aim, split-second movement decisions, and pre-round communication. A snapshot records that CT has more armor and money, but not whether their entry fragger is on a hot streak.

**Random Forest edges out XGBoost** (76.0% vs. 75.4%), which is somewhat surprising given XGBoost's reputation as the dominant tabular model. The explanation likely lies in feature correlations: the 34-per-side weapon columns are highly correlated (most are zero; a few signal "rifle round" or "eco round"). RF's random subsampling at each split (`max_features=0.5`) decorrelates the trees and prevents any single weapon feature from dominating. XGBoost's sequential correction, by contrast, may overfit to noise in these sparse columns.

**Logistic Regression achieves 74.9%**, only 1.1pp below the best model. This reveals the data is **substantially linearly separable**: more health, armor, and money correlates monotonically with winning. The optimal C=0.01 (strong regularization) shrinks uninformative features—most weapon columns—and prevents overfitting, suggesting the useful signal is concentrated in a small number of features rather than spread across all 96.

**Neural Network ties LR** despite its greater capacity. The dataset lacks the spatial or sequential structure where deep learning typically excels; the tabular snapshot format means all predictive information is already explicit in the feature values. Adding non-linear depth adds model complexity without accessing additional signal.

The **CT recall asymmetry** (CT: 0.82, T: 0.71) is structurally meaningful. CT teams defend sites with rifles, typically from better-armored positions with more predictable win conditions (hold the site or kill all Ts). T-side wins often require breaking through a defensive setup via specific executes or split pushes—actions not visible in a single snapshot—making them harder to predict.

### 5.2 RQ2: The Three-Phase Structure

The 12.9pp accuracy gap between mid-round (68.5%) and late-round (81.5%) is the strongest finding in this study, revealing that prediction difficulty is not uniform but follows a **three-phase structure** driven by which information is available.

**Early-round (76.4%)** accuracy is surprisingly high for a period when no combat has occurred. Feature importance explains this: economic features account for 74.9% of importance, with armor and money dominating. This reflects **economic inertia from previous rounds**: the team that won the pistol round starts the next round with rifles; the loser buys cheap weapons or saves entirely. The model is effectively predicting buy-phase outcomes from the carry-over of prior round results—not the current round itself. The fact that `time_left` ranks in the top 10 for early rounds (but drops out later) suggests the model also learns sub-phases within the "early" window.

**Mid-round (68.5%)** is the hardest because the game state is most uncertain: both teams are at or near full strength, combat is actively occurring, and individual skill dictates who survives. The 7.8pp accuracy drop relative to early-round directly quantifies the contribution of tactical uncertainty—information no snapshot can encode.

**Late-round (81.5%)** becomes easy again because combat has largely resolved. Survival features (health, armor, players alive) surge to 49.3% importance—nearly equal to economic features—because they directly reflect who is winning the firefight. With 37.4% of late snapshots having the bomb planted, the tactical situation is also highly constrained: the round becomes a binary defuse problem where time and player counts are decisive.

The practical implication is that a single accuracy figure for round prediction is misleading: a real-time system should assign lower confidence to mid-round predictions and higher confidence to late-round ones. Phase-specific models or confidence-weighting would be a natural extension.

### 5.3 RQ3: Where Do Models Fail — Uncertainty or Structural Surprise?

The central finding of RQ3 is that **model errors are not uniformly distributed across game states—they are heavily concentrated in comeback situations**, where the structurally disadvantaged team wins. The 72.5% error rate in comeback rounds is three times the overall error rate (23.8%) and almost eighteen times lower than in one-sided rounds (4.3%).

**One-sided rounds (95.7% accuracy)** confirm that the model captures observable advantages reliably. When one team leads in players alive, money, and health simultaneously, the snapshot is genuinely informative and the model exploits it correctly. This validates the core feature design: the combination of economic and survival signals is sufficient to predict expected outcomes.

**Close rounds (73.3% accuracy)** perform at roughly the overall average. Structurally balanced situations are hard to predict, but not catastrophically so—the model hedges appropriately, as evidenced by 37.8% of close-round predictions falling in the uncertain confidence band (0.4–0.6). This is the expected behaviour of a well-calibrated model under genuine ambiguity.

**Comeback rounds (27.5% accuracy) reveal the fundamental limit of snapshot-based prediction.** The model achieves below-chance performance—worse than a coin flip—because it systematically predicts the favoured team and the favoured team loses. Critically, the confidence distribution for comeback rounds (Table 6) mirrors that of close rounds: 42% of predictions fall in the 0.4–0.6 band, and the mean P(CT wins) is exactly 0.500. The model is not blindly overconfident in comeback rounds; it simply cannot detect the signal that an upset is about to occur.

This is the key distinction: **errors in close rounds stem from genuine uncertainty** (the model knows it doesn't know), while **errors in comeback rounds stem from structural surprise** (the snapshot looks clear, but the outcome defies it). The factors driving upsets—individual skill, communication, positional reads—are entirely absent from the tabular snapshot. A model that correctly identifies its own ignorance would need access to player-level historical performance or real-time positional data, neither of which is available in this dataset.

The practical implication is that a real-time prediction system built on this data should be most trusted in one-sided situations and treated with caution even when it produces high-confidence predictions—because high confidence in comeback rounds does not mean the prediction is correct.

---

## 6. Conclusion

This study investigated CS:GO round winner prediction from per-snapshot tabular features, addressing three research questions.

**RQ1** established that all four model families (LR, NN, XGBoost, RF) cluster within a 1.1pp accuracy range (74.9%–76.0%), with Random Forest performing best at 76.0%. This narrow band reflects a data ceiling: the irreducible uncertainty from player actions not captured in snapshots. Logistic Regression's competitive performance indicates the data is substantially linearly separable, while the Neural Network's failure to exceed LR confirms that depth adds no benefit on this tabular, non-sequential dataset.

**RQ2** revealed a three-phase prediction structure—economic (early, 76.4%), chaotic (mid, 68.5%), survival (late, 81.5%)—with a 12.9pp accuracy gap between the easiest and hardest phases. Feature importance shifts quantitatively confirm this structure: economic features dominate early and mid rounds, while survival features rise to near-parity in late rounds. This finding suggests real-time prediction systems should vary confidence levels based on round phase.

**RQ3** revealed that model errors are heavily concentrated in comeback situations (72.5% error rate) rather than close rounds (26.7%), despite both having similar model confidence distributions. The model fails not because it is uncertain, but because snapshot data cannot encode the individual-skill factors that drive upsets. One-sided rounds, by contrast, are predicted with 95.7% accuracy, confirming the model reliably captures observable team advantages. This asymmetry shows that overall accuracy masks fundamentally different prediction regimes across game states.

The main limitation is the snapshot-based approach: individual skill, positioning, and communication are invisible to the model. Future work incorporating time-series of snapshots or player-level performance histories could directly address the mid-round prediction gap.

---

## Generative AI Usage

Claude (Anthropic) was used in this project to assist with:
- Structuring and refactoring Python code into modular RQ-specific scripts
- Drafting and editing sections of this report
- Debugging model training errors

All experimental design, research question formulation, result interpretation, and analytical conclusions were produced by the team. Claude was not used to generate or extend the dataset.

---

## Bibliography

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

Lillelund, C. (2020). *CS:GO Round Winner Classification* [Dataset]. Kaggle. https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification

Semenov, A., Romov, P., Korolev, S., Yashkov, D., & Neklyudov, K. (2016). Performance of machine learning algorithms in predicting game outcome from drafts in Dota 2. In *International Conference on Analysis of Images, Social Networks and Texts* (pp. 26–37). Springer, Cham.

Xenopoulos, P., Coelho, B., & Silva, C. (2020). Valuing actions in Counter-Strike: Global Offensive. In *2020 IEEE International Conference on Big Data* (pp. 1283–1292). IEEE.
