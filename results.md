# CS:GO Round Winner — Experiment Results

## RQ1: Model Comparison

### Summary Table

| Model | Acc (Baseline) | Acc (Tuned) | F1 (Baseline) | F1 (Tuned) |
|---|---|---|---|---|
| **Random Forest** | 0.7559 | **0.7602** | 0.7559 | **0.7597** |
| XGBoost | 0.7530 | 0.7543 | 0.7530 | 0.7541 |
| Logistic Regression | 0.7482 | 0.7489 | 0.7483 | 0.7490 |
| Neural Network | 0.7478 | 0.7496 | 0.7478 | 0.7496 |

> Baseline = default params (RF: 100 trees; XGBoost: default; LR: default; NN: lr=0.001, dropout=0.3, hidden=128)

### Best Hyperparameters

| Model | Best Params | Best CV Acc |
|---|---|---|
| Random Forest | n_estimators=500, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features=0.5 | — (pre-set) |
| XGBoost | subsample=0.7, reg_alpha=0, n_estimators=300, min_child_weight=1, max_depth=5, learning_rate=0.05, colsample_bytree=0.8 | 0.7505 |
| Logistic Regression | C=0.01, solver=lbfgs | 0.7472 |
| Neural Network | learning_rate=0.001, dropout=0.2, hidden_dim=128 | val_acc=0.7510 |

### Classification Reports (Tuned Models)

**Random Forest**
| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CT | 0.72 | 0.82 | 0.77 | 11799 |
| T  | 0.81 | 0.71 | 0.75 | 12658 |
| accuracy | | | 0.76 | 24457 |

**XGBoost**
| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CT | 0.72 | 0.80 | 0.76 | 11799 |
| T  | 0.79 | 0.72 | 0.75 | 12658 |
| accuracy | | | 0.75 | 24457 |

**Logistic Regression**
| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CT | 0.73 | 0.76 | 0.75 | 11799 |
| T  | 0.77 | 0.74 | 0.75 | 12658 |
| accuracy | | | 0.75 | 24457 |

**Neural Network**
| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CT | 0.73 | 0.76 | 0.74 | 11799 |
| T  | 0.77 | 0.74 | 0.75 | 12658 |
| accuracy | | | 0.75 | 24457 |

### NN Hyperparameter Search (12 combinations)

| learning_rate | dropout | hidden_dim | val_acc |
|---|---|---|---|
| **0.001** | **0.2** | **128** | **0.7510** ✓ |
| 0.001 | 0.2 | 256 | 0.7401 |
| 0.001 | 0.3 | 128 | 0.7478 |
| 0.001 | 0.3 | 256 | 0.7399 |
| 0.001 | 0.4 | 128 | 0.7452 |
| 0.001 | 0.4 | 256 | 0.7450 |
| 0.005 | 0.2 | 128 | 0.7448 |
| 0.005 | 0.2 | 256 | 0.7419 |
| 0.005 | 0.3 | 128 | 0.7454 |
| 0.005 | 0.3 | 256 | 0.7400 |
| 0.005 | 0.4 | 128 | 0.7476 |
| 0.005 | 0.4 | 256 | 0.7420 |

NN Training loss: 0.4362 → 0.3883 (epoch 5→30)

---

## RQ2: Temporal Analysis

### Segment Sizes

| Segment | time_left | Rows | Bomb Planted Rate |
|---|---|---|---|
| Early | > 120s | 33,715 | 0.0% |
| Mid | 60–120s | 52,064 | 0.0% |
| Late | < 60s | 36,631 | 37.4% |

### Accuracy by Phase

| Segment | Accuracy | F1 (weighted) |
|---|---|---|
| Early (>120s) | 0.7643 | 0.7644 |
| **Mid (60–120s)** | **0.6852** | **0.6833** |
| **Late (<60s)** | **0.8147** | **0.8143** |

### Classification Reports

**Early (>120s)**
| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CT | 0.78 | 0.77 | 0.77 | 3700 |
| T  | 0.75 | 0.76 | 0.76 | 3458 |
| accuracy | | | 0.76 | 7158 |

**Mid (60–120s)**
| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CT | 0.68 | 0.76 | 0.71 | 5438 |
| T  | 0.70 | 0.61 | 0.65 | 5020 |
| accuracy | | | 0.69 | 10458 |

**Late (<60s)**
| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CT | 0.80 | 0.77 | 0.79 | 3194 |
| T  | 0.83 | 0.85 | 0.84 | 4081 |
| accuracy | | | 0.81 | 7275 |

### Feature Importance by Phase

| Segment | Accuracy | Economic share | Survival share | Other |
|---|---|---|---|---|
| Early (>120s) | 0.7657 | 74.9% | 22.8% | 2.3% |
| Mid (60–120s) | 0.6852 | 71.7% | 26.1% | 2.2% |
| Late (<60s)   | 0.8136 | 49.2% | **49.3%** | 1.4% |

### Top 10 Features per Segment

**Early (>120s)**
| Feature | Importance | Category |
|---|---|---|
| t_armor | 0.1089 | survival |
| ct_armor | 0.0837 | survival |
| ct_money | 0.0823 | economic |
| t_money | 0.0810 | economic |
| t_helmets | 0.0741 | economic |
| ct_defuse_kits | 0.0363 | economic |
| t_weapon_ak47 | 0.0343 | economic |
| ct_helmets | 0.0330 | economic |
| time_left | 0.0326 | survival |
| t_grenade_flashbang | 0.0294 | economic |

**Mid (60–120s)**
| Feature | Importance | Category |
|---|---|---|
| ct_armor | 0.0743 | survival |
| t_armor | 0.0621 | survival |
| ct_money | 0.0463 | economic |
| t_money | 0.0449 | economic |
| t_helmets | 0.0421 | economic |
| ct_helmets | 0.0355 | economic |
| ct_defuse_kits | 0.0346 | economic |
| t_health | 0.0341 | survival |
| ct_health | 0.0308 | survival |
| ct_grenade_flashbang | 0.0294 | economic |

**Late (<60s)**
| Feature | Importance | Category |
|---|---|---|
| ct_armor | 0.0871 | survival |
| t_armor | 0.0864 | survival |
| t_health | 0.0761 | survival |
| ct_health | 0.0703 | survival |
| t_players_alive | 0.0664 | survival |
| t_helmets | 0.0538 | economic |
| t_money | 0.0434 | economic |
| ct_players_alive | 0.0429 | survival |
| bomb_planted | 0.0347 | survival |
| ct_money | 0.0340 | economic |

---

## RQ3: Feature Engineering

### Feature Subset Comparison (Selective Diffs)

| Feature Set | # Features | Accuracy | F1 |
|---|---|---|---|
| All features (raw + engineered) | 108 | 0.7577 | 0.7575 |
| Survival only | 11 | 0.7409 | 0.7407 |
| Economic only | 13 | 0.7352 | 0.7352 |

### Top 20 Feature Importances (All Features model)

| Rank | Feature | Importance | Category |
|---|---|---|---|
| 1 | armor_diff | 0.1032 | survival |
| 2 | weapon_value_diff | 0.0665 | economic |
| 3 | health_diff | 0.0488 | survival |
| 4 | helmets_diff | 0.0487 | economic |
| 5 | players_alive_diff | 0.0431 | survival |
| 6 | ct_armor | 0.0429 | survival |
| 7 | ct_weapon_value | 0.0418 | economic |
| 8 | t_armor | 0.0383 | survival |
| 9 | t_weapon_value | 0.0354 | economic |
| 10 | money_diff | 0.0326 | economic |
| 11 | t_money | 0.0303 | economic |
| 12 | ct_money | 0.0296 | economic |
| 13 | grenade_diff | 0.0274 | economic |
| 14 | time_left | 0.0222 | other |
| 15 | t_helmets | 0.0215 | economic |
| 16 | t_score | 0.0175 | other |
| 17 | ct_total_grenades | 0.0173 | economic |
| 18 | ct_score | 0.0171 | other |
| 19 | t_total_grenades | 0.0161 | economic |
| 20 | t_health | 0.0158 | survival |

### Diff-ification Strategy Comparison

| Strategy | # Features | Accuracy | F1 |
|---|---|---|---|
| Raw only | 96 | **0.7623** | **0.7621** |
| Diff only | 52 | 0.7564 | 0.7562 |
| Raw + All diffs | 145 | 0.7570 | 0.7568 |

### Top 10 Features per Strategy

**Raw only**
| Feature | Importance |
|---|---|
| ct_armor | 0.0840 |
| t_armor | 0.0840 |
| t_helmets | 0.0583 |
| t_money | 0.0479 |
| ct_money | 0.0478 |
| ct_helmets | 0.0391 |
| t_health | 0.0376 |
| ct_defuse_kits | 0.0373 |
| ct_health | 0.0351 |
| time_left | 0.0317 |

**Diff only**
| Feature | Importance |
|---|---|
| diff_armor | 0.1627 |
| diff_weapon_value | 0.0954 |
| diff_helmets | 0.0739 |
| diff_health | 0.0664 |
| ct_weapon_value | 0.0643 |
| diff_players_alive | 0.0557 |
| diff_money | 0.0541 |
| t_weapon_value | 0.0502 |
| time_left | 0.0366 |
| diff_score | 0.0289 |

**Raw + All diffs**
| Feature | Importance |
|---|---|
| diff_armor | 0.1018 |
| diff_weapon_value | 0.0603 |
| diff_helmets | 0.0437 |
| diff_health | 0.0421 |
| diff_players_alive | 0.0407 |
| ct_armor | 0.0404 |
| t_armor | 0.0378 |
| ct_weapon_value | 0.0375 |
| t_weapon_value | 0.0348 |
| diff_money | 0.0286 |
