# Predicting Round Outcomes in Counter-Strike: Global Offensive Using Machine Learning

**Group ID:** [Your Group ID]  
**Members:** [Member 1], [Member 2], [Member 3]  
**COMP90049 Introduction to Machine Learning, Semester 1 2026**

---

## Abstract

In this project, we use machine learning to predict which team wins a round in Counter-Strike: Global Offensive (CS:GO), based on a snapshot of the game state taken during the round. We compare four classifiers — Logistic Regression, Random Forest, XGBoost, and a Neural Network — and find that Random Forest performs best with 76.2% accuracy after hyperparameter tuning. We also investigate whether the timing of the snapshot affects how hard the prediction is. Interestingly, mid-round snapshots (60–120 seconds remaining) are the hardest to predict (68.5%), while late-round snapshots are the easiest (81.5%). By looking at feature importances across time segments, we find that early-round prediction relies heavily on economic features like money and equipment, while late-round prediction shifts toward survival features like health and players alive. We discuss why this happens and what it means for the limits of snapshot-based prediction.

---

## 1. Introduction

CS:GO is a competitive team-based shooter where two sides — Counter-Terrorists (CT) and Terrorists (T) — play a series of rounds. Each round involves buying weapons with in-game money, executing tactical strategies, and trying to eliminate the other team or complete the round objective (planting or defusing a bomb). The outcome of each round is binary: either CT wins or T wins.

We chose this problem because it is a concrete, well-defined classification task with a rich dataset and a real-world application. Being able to predict round outcomes mid-round could be useful for broadcasting (live win probability displays), coaching analysis, or spectator tools. It is also interesting from a machine learning perspective because the dataset is quite large and contains many different types of features — health, money, weapons, grenades — which makes it suitable for comparing different types of models.

We address two research questions in this project:

- **RQ1:** Which machine learning model performs best at predicting round winners, and what does the comparison tell us about the nature of this prediction problem?
- **RQ2:** Does the timing of a snapshot within a round affect how accurately we can predict the winner? If so, why?

RQ1 is our main question and involves comparing four models with proper hyperparameter tuning. RQ2 came out of a curiosity we had while looking at the results — we wondered if some snapshots are inherently harder to predict than others. We think the combination of these two questions gives a more complete picture of the problem than just reporting model accuracy numbers.

The dataset we use is the CS:GO Round Winner Classification dataset from Kaggle \citep{lillelund2020csgo}. It contains 122,410 snapshots from 23,537 rounds across seven maps. We chose this dataset because it is large enough to train and compare multiple models reliably, the class distribution is nearly balanced (49% CT wins, 51% T wins), and the features are meaningful and interpretable from a game perspective.

---

## 2. Literature Review

The dataset we use was published by \citet{lillelund2020csgo} on Kaggle. It does not come with a research paper, but it has been widely used in the community for classification experiments. We found it through Kaggle's dataset search and chose it because of its size and the variety of features it provides.

\citet{xenopoulos2020valuing} studied how to quantify the value of individual actions in CS:GO, such as kills and positions. Their work is relevant to ours because they found that player positions and kill events are among the strongest predictors of round outcomes — but these features are not present in our snapshot dataset. This helped us understand why our models have a performance ceiling, since we are missing some of the most important information.

\citet{hodge2019win} compared different machine learning models for predicting win outcomes across several esports games, including MOBA titles. They found that gradient boosting methods (like XGBoost) tend to outperform neural networks on structured tabular esports data. This was one reason we included both XGBoost and Random Forest as our main candidates in RQ1, and it is interesting to compare whether their finding holds in the CS:GO setting.

\citet{chen2016xgboost} introduced XGBoost, one of the models we use. We include this reference because understanding how XGBoost handles sparse features (most weapon columns in our dataset are zero for most rounds) helped us understand why it performed the way it did in our experiments.

---

## 3. Method

### 3.1 Dataset Description

The dataset has 122,410 rows and 96 features. Each row is a snapshot of the game state at a specific moment in a round. The features include:

- Health and armour for both sides
- Number of players alive on each side
- Cash on hand for each side
- Count of each weapon type held by each side (34 weapon types)
- Count of each grenade type held (6 types)
- Time remaining in the round (`time_left`, ranges from 0 to 175 seconds)
- Whether the bomb has been planted
- The map name and current round scores

The target variable is `round_winner`, which is either "CT" or "T". As mentioned above, the class distribution is nearly balanced (49/51), so we did not need to apply any resampling.

One thing we noticed about the data is that most of the weapon-related features are zero for most rows — players do not carry every weapon at once. This makes the feature space quite sparse, which we kept in mind when choosing models.

### 3.2 Data Preprocessing

We applied the following preprocessing steps:

- **Label encoding:** The `map` column (7 unique values) was converted to integers using `LabelEncoder`. The `bomb_planted` column (boolean) was converted to 0/1.
- **Standardisation:** We applied `StandardScaler` (zero mean, unit variance) to all features. This is important for models like LR and KNN that are sensitive to feature scale. Tree-based models (RF, XGBoost) do not require scaling, but we applied it consistently across all models for a fair comparison.

### 3.3 Avoiding Data Leakage

This was an important issue we discovered early on. Because each round generates multiple snapshots (usually 4–6), all with the same `round_winner` label, a random 80/20 train/test split would allow the model to see snapshots from the same round in both training and test sets. This would artificially inflate accuracy because the model can essentially "recognise" rounds it has already partially seen.

To fix this, we identified round boundaries by detecting when `time_left` increases from one row to the next (which signals the start of a new round). We assigned each snapshot a `round_id` and then used `GroupShuffleSplit` to ensure all snapshots from a given round end up in either training or test — never both. This gave us 23,537 unique rounds, with about 18,830 rounds for training (~97,960 snapshots) and 4,707 rounds for testing (~24,450 snapshots).

We applied the same group-based logic for 3-fold cross-validation during hyperparameter tuning, since the same leakage problem would apply there as well.

### 3.4 Feature Engineering

Looking at the original features, we noticed that the dataset represents each side's state separately (e.g., `ct_armor` and `t_armor` as two columns). But in a game like CS:GO, what usually matters is not how much armour CT has in absolute terms, but whether CT has *more* armour than T. This led us to create a set of difference features:

| Feature | Formula |
|---|---|
| `health_diff` | `ct_health − t_health` |
| `armor_diff` | `ct_armor − t_armor` |
| `players_alive_diff` | `ct_players_alive − t_players_alive` |
| `money_diff` | `ct_money − t_money` |
| `helmets_diff` | `ct_helmets − t_helmets` |
| `ct_weapon_value` | Sum of (weapon count × weapon price) for CT |
| `t_weapon_value` | Sum of (weapon count × weapon price) for T |
| `weapon_value_diff` | `ct_weapon_value − t_weapon_value` |
| `grenade_diff` | CT total grenades − T total grenades |

For weapon values, we used approximate in-game prices (e.g., AK-47: \$2,700, AWP: \$4,750). We thought aggregating by price would be more informative than individual weapon columns, since what matters economically is the total firepower investment, not which specific weapon a player holds.

We also tried creating difference features for *every* paired column (all 44 ct\_/t\_ pairs), to see if more difference features would help. Interestingly, this actually reduced accuracy slightly (75.6% vs. 76.2%), which we discuss in Section 5.

### 3.5 Models

We chose five models that represent different approaches to classification:

- **Logistic Regression (LR):** A simple linear model, useful as a baseline to see if the data is roughly linearly separable.
- **Random Forest (RF):** An ensemble of decision trees that uses bagging and feature subsampling to reduce overfitting.
- **XGBoost:** A gradient boosting method that builds trees sequentially, with regularisation to prevent overfitting. It also handles sparse features efficiently.
- **Neural Network (NN):** A three-hidden-layer network with BatchNorm, ReLU activations, and Dropout. We included this to test whether deep learning adds value on this type of tabular data compared to tree-based methods.

### 3.6 Hyperparameter Tuning

We tuned each model using either GridSearchCV or RandomizedSearchCV with 3-fold cross-validation. The NN was tuned manually using a held-out 15% validation split, since it is harder to integrate into sklearn's CV pipeline.

| Model | Hyperparameter | Search Range | Best Value |
|---|---|---|---|
| LR | C (regularisation strength) | [0.01, 0.1, 1.0, 10.0] | **0.01** |
| LR | solver | [lbfgs, saga] | **lbfgs** |
| RF | n\_estimators | [100, 200, 300, 500] | **500** |
| RF | max\_depth | [10, 20, 30, None] | **20** |
| RF | min\_samples\_split | [2, 5, 10] | **5** |
| XGBoost | max\_depth | [3, 5, 6, 9] | **5** |
| XGBoost | learning\_rate | [0.01, 0.05, 0.1, 0.3] | **0.01** |
| XGBoost | n\_estimators | [100, 200, 300, 500] | **300** |
| XGBoost | subsample | [0.6, 0.8, 1.0] | **0.6** |
| XGBoost | colsample\_bytree | [0.6, 0.8, 1.0] | **1.0** |
| NN | learning\_rate | [0.001, 0.005] | **0.001** |
| NN | dropout rate | [0.2, 0.3, 0.4] | **0.4** |
| NN | hidden layer size | [128, 256] | **128** |

### 3.7 Evaluation

We report accuracy, precision, recall, and weighted F1-score for all models. Since the classes are nearly balanced, accuracy is a reasonable primary metric. We also include a confusion matrix for the best model to understand what kinds of mistakes it makes. For RQ2, we compare per-segment accuracy across early, mid, and late round phases, and use feature importances from the RF model to help explain the differences.

---

## 4. Results

### 4.1 RQ1: Which Model Performs Best?

Table 1 and Figure 1 show the results before and after tuning.

**Table 1: Model performance before and after hyperparameter tuning**

| Model | Before Tuning | After Tuning | Precision | Recall | F1 | Best Parameters |
|---|---|---|---|---|---|---|
| **Random Forest** | 75.77% | **76.20%** | 0.762 | 0.762 | **0.762** | n\_est=500, depth=20 |
| XGBoost | 75.90% | 75.39% | 0.754 | 0.754 | 0.754 | lr=0.01, depth=5 |
| Neural Network | 74.85% | 74.89% | 0.749 | 0.749 | 0.749 | lr=0.001, drop=0.4 |
| Logistic Regression | 74.82% | 74.89% | 0.749 | 0.749 | 0.749 | C=0.01 |

Overall, Random Forest is the best model at 76.2%. The differences between models are not huge — they all cluster in the 74–76% range except KNN. Figure 4 shows the confusion matrix for RF. We can see that CT wins are classified more accurately (precision 80%) than T wins (73%), which we discuss in Section 5.

Looking at feature importances (Figure 3), the top features are all difference features we engineered: `armor_diff` (10.3%), `weapon_value_diff` (6.7%), `health_diff` (4.9%), `helmets_diff` (4.9%), and `players_alive_diff` (4.3%). This suggests our feature engineering was useful — the relative difference between teams matters more than each team's absolute values.

### 4.2 RQ2: Does Timing Matter?

We split the data into three segments based on `time_left` and evaluated RF on each one separately:

**Table 2: Accuracy by round phase and feature importance category share**

| Segment | Rows | Bomb Planted | Accuracy | F1 | Economic Importance | Survival Importance |
|---|---|---|---|---|---|---|
| Early (>120s) | 33,715 | 0.0% | 76.52% | 0.765 | **75.5%** | 22.0% |
| Mid (60–120s) | 52,064 | 0.0% | 68.54% | 0.684 | 71.9% | 25.8% |
| Late (<60s) | 36,631 | 37.4% | **81.47%** | 0.815 | 50.6% | **47.8%** |

The results surprised us. We expected accuracy to go up as the round progresses (more things have happened, so there should be more information). But what we actually see is a U-shape: early is moderate (76.5%), mid drops significantly (68.5%), and late jumps to 81.5%. Figure 2 shows this pattern, along with how the feature importance shifts across phases.

---

## 5. Discussion

### 5.1 RQ1: What the Model Comparison Tells Us

Random Forest being the best model makes sense to us in hindsight. It is an ensemble of many trees (500 in the optimal configuration), which helps average out the noise from the many sparse weapon features. The depth limit of 20 gives enough capacity to capture non-linear interactions (e.g., the combination of armour + health + players alive is probably more predictive than any one feature alone) without overfitting.

We were a bit surprised that XGBoost performed slightly *worse* after tuning (75.9% → 75.4%), since XGBoost usually does well on tabular data \citep{hodge2019win}. Looking at the best parameters, we think the issue is that RandomizedSearch only covered 15 combinations out of a large search space, and the configuration it found (learning rate=0.01 with only 300 trees) is probably underfitting — a slow learning rate needs more trees to fully converge. If we had more compute time to search more combinations, XGBoost might catch up to RF.

LR and NN ending up at the same accuracy (74.9%) is an interesting result. For LR, the best regularisation strength was C=0.01 (the strongest we tested), which means the model is heavily shrinking most feature weights toward zero. This makes sense given how many of the weapon count features are nearly always zero — there is not much signal there. For NN, we expected it to do better given its capacity to learn non-linear relationships, but it ended up no better than the linear model. We think this is because tabular data without any spatial or sequential structure does not naturally benefit from deep learning — a finding consistent with \citet{hodge2019win}.

KNN improved the most from tuning (+1.78pp). Switching from euclidean to manhattan distance helped because, with 96 features most of which are zero, euclidean distance between any two points becomes very similar (everything is "far" in a similar way). Manhattan distance is less affected by this problem and gave more useful neighbours.

One methodological finding worth highlighting is the data leakage issue. Before we fixed the splitting strategy, all models appeared about 1–2pp better. This seems small, but it represents an inflated metric that would not reflect real-world performance, so we consider fixing it essential.

We also noticed that an existing public project using the same dataset \citep{joyoadikusumo2021csgo} reported 88.41% accuracy with Random Forest — significantly higher than our 76.2%. After examining their code, we believe this gap is largely explained by two sources of leakage. First, they used a standard random `train_test_split` without grouping by round, which allows snapshots from the same round to appear in both training and test sets. Since all snapshots within a round share the same label, the model can effectively memorise round outcomes it has partially seen. Second, their K-means clustering feature was fitted on the full dataset before splitting, so the cluster assignments used as training features were already influenced by the test data. Our group-based split prevents both issues and gives a more realistic estimate of how well a model would generalise to genuinely unseen rounds.

Regarding feature engineering: we found that selectively computing differences for meaningful pairs (armour, health, players alive, money, weapon value) improved the model, but creating differences for *all* paired columns did not help. One reason might be that raw features contain useful information about each side individually — for example, CT players tend to buy armour more consistently than T, so `ct_armor` alone still carries some predictive signal that `armor_diff` does not fully capture.

### 5.2 RQ2: Why Does Timing Matter So Much?

The U-shape in accuracy across round phases (76.5% → 68.5% → 81.5%) was the most interesting finding in our project. We think it can be explained by what kind of information is available at each phase.

**Early phase (>120s remaining):** At the start of a round, all players are alive and nobody has fired yet. In terms of survival features, there is almost nothing to distinguish the two teams — both have full health. But the economic features are quite different: money on hand, weapons purchased, and equipment reflect the *outcome of previous rounds*. Teams that have been winning recent rounds tend to have better weapons and more money. So the model can predict the winner based on economic advantage inherited from previous rounds. This explains why economic features account for 75.5% of importance in this segment.

**Mid phase (60–120s remaining):** This is where things get harder. Both teams have engaged — some players may have died, some weapons may have been picked up — but the round is not resolved yet. A lot of what determines the outcome at this point is how well individual players are performing: their aim, their positioning, their communication. None of these appear in the snapshot. This is probably why accuracy drops to 68.5% in the mid phase — the model is essentially guessing based on incomplete information while the most important events are unfolding.

**Late phase (<60s remaining):** By now, the round is close to being resolved. Player counts have often diverged significantly, and 37.4% of snapshots have the bomb planted, which changes the game state dramatically (CT must defuse, T defends). Survival features like `t_health`, `ct_health`, and `players_alive` become much more discriminating because they directly reflect who has been winning the combat exchange. This explains why survival feature importance jumps to 47.8% and accuracy reaches 81.5%.

We also notice that T wins are consistently harder to predict than CT wins (Figure 4 shows that the false negative rate for T is higher). A possible explanation is that T has to actively take map control and make aggressive moves, which are harder to predict from a static snapshot than CT's more defensive positioning.

To try to improve mid-round accuracy, we experimented with full diff-ification of features, but it did not help. The fundamental limitation seems to be the absence of positional and event data, which cannot be addressed through feature engineering on the existing columns.

---

## 6. Conclusion

In this project, we trained and compared five machine learning models to predict CS:GO round winners from game state snapshots, and investigated how prediction difficulty changes across the round.

For **RQ1**, Random Forest achieved the best accuracy of 76.2% after tuning, slightly ahead of XGBoost (75.4%). Both outperform the linear model (LR, 74.9%) and the neural network (74.9%), which confirms that tree-based ensemble methods suit this type of sparse tabular data well. Our feature engineering — particularly difference features like `armor_diff` and `weapon_value_diff` — turned out to be important, as these consistently ranked as the most informative features.

For **RQ2**, we found that mid-round snapshots are significantly harder to predict (68.5%) than early-round (76.5%) or late-round (81.5%) snapshots. This surprised us initially, but makes sense when looking at the feature importances: early-round prediction relies on economic state (inherited from previous rounds), while late-round prediction relies on survival state (who is alive, health levels). The mid phase is the most uncertain because the factors that determine outcome — individual skill and positioning — are not captured in the data.

Overall, we think the prediction ceiling of around 76% on the full dataset is largely a result of missing information rather than model limitations. Incorporating player positions or sequences of snapshots rather than single snapshots could likely push accuracy meaningfully higher, and would be an interesting direction for future work.

---

## Generative AI Usage

We used Claude (Anthropic, claude-sonnet-4-6) as an assistant throughout this project. Specifically:
- **Brainstorming:** We discussed research question ideas and data analysis approaches with Claude early in the project.
- **Code:** Claude helped write boilerplate code for model training, hyperparameter search, and figure generation. All code was reviewed and tested by team members before use.
- **Writing:** Claude produced initial drafts of some report sections, which were then rewritten and edited substantially by the team to reflect our own understanding and voice.

All experimental results in this report were produced by code that we ran and verified ourselves.

---

## Bibliography

\bibitem{lillelund2020csgo}
Christian Lillelund. 2020. *CS:GO Round Winner Classification*. Kaggle Dataset. \url{https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification}

\bibitem{xenopoulos2020valuing}
Peter Xenopoulos, Jonathon Dorsey, Claudio Silva, and Brian Hilburn. 2020. Valuing actions in Counter-Strike: Global Offensive. In *2020 IEEE International Conference on Big Data (Big Data)*, pages 1891–1898. IEEE.

\bibitem{hodge2019win}
Victoria J. Hodge, Sam Devlin, Nick Sephton, Florian Block, Peter Cowling, and Anders Drachen. 2019. Win prediction in esports: Mixed-rank match prediction in multi-player online battle arena games. *arXiv preprint arXiv:1911.09631*.

\bibitem{chen2016xgboost}
Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pages 785–794.

\bibitem{joyoadikusumo2021csgo}
Ananto Joyoadikusumo. 2021. *CS:GO Round Winner Classification*. GitHub Repository. \url{https://github.com/anantoj/csgo-round-winner-classification}
