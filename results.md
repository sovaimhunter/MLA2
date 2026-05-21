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
| Random Forest | n_estimators=500, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features=0.5 | 0.7502 |
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

## RQ3: Model Error Analysis by Round Situation

> Overall acc = 76.2%  |  Model: Random Forest (n_estimators=500, max_depth=20)

### Situation Distribution & Accuracy

| Situation | N | Share | Accuracy | Error Rate |
|---|---|---|---|---|
| Close | 16,655 | 68.1% | 73.3% | 26.7% |
| One-Sided | 6,279 | 25.7% | **95.7%** | 4.3% |
| Comeback | 1,523 | 6.2% | 27.5% | **72.5%** |

### Model Confidence Distribution (P(CT wins))

| Situation | Low conf (<0.4) | Uncertain (0.4–0.6) | High conf (>0.6) |
|---|---|---|---|
| Close | 31% | 38% | 31% |
| One-Sided | 47% | 12% | 41% |
| Comeback | 29% | **42%** | 29% |
