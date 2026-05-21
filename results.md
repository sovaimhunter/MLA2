# CS:GO Round Winner Classifier — 训练结果

## 运行环境

- Python 3.14.5
- XGBoost 3.2.0 / scikit-learn 1.8.0 / PyTorch（GPU 检测自动）
- GPU：未检测到 CUDA，所有模型均使用 CPU 运行

---

## 模型结果

### XGBoost

| 指标 | 值 |
|------|----|
| Test Accuracy | **75.42%** |
| Best CV Accuracy | 75.00% |

**最优参数：**
```
subsample=0.7, reg_alpha=0, n_estimators=300,
min_child_weight=5, max_depth=5,
learning_rate=0.05, colsample_bytree=1.0
```

**分类报告：**

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|----|---------|
| CT | 0.72 | 0.80 | 0.76 | 11799 |
| T  | 0.79 | 0.71 | 0.75 | 12658 |
| **accuracy** | | | **0.75** | 24457 |

---

### Logistic Regression

| 指标 | 值 |
|------|----|
| Test Accuracy | **74.89%** |
| Best CV Accuracy | 74.72% |

**最优参数：**
```
C=0.01, solver=lbfgs
```

**分类报告：**

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|----|---------|
| CT | 0.73 | 0.76 | 0.75 | 11799 |
| T  | 0.77 | 0.74 | 0.75 | 12658 |
| **accuracy** | | | **0.75** | 24457 |

---

### Random Forest

| 指标 | 值 |
|------|----|
| Test Accuracy | **76.06%** |
| Best CV Accuracy | 75.02% |

**最优参数：**
```
n_estimators=500, min_samples_split=2, min_samples_leaf=1,
max_features=0.5, max_depth=10
```

**分类报告：**

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|----|---------|
| CT | 0.72 | 0.82 | 0.77 | 11799 |
| T  | 0.81 | 0.71 | 0.75 | 12658 |
| **accuracy** | | | **0.76** | 24457 |

---

### KNN

| 指标 | 值 |
|------|----|
| Test Accuracy | **73.97%** |
| Best CV Accuracy | 73.07% |

**最优参数：**
```
metric=manhattan, n_neighbors=21
```

**分类报告：**

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|----|---------|
| CT | 0.72 | 0.76 | 0.74 | 11799 |
| T  | 0.76 | 0.72 | 0.74 | 12658 |
| **accuracy** | | | **0.74** | 24457 |

---

### Neural Network

| 指标 | 值 |
|------|----|
| Test Accuracy | **75.08%** |
| Best Val Accuracy | 74.90% |

**最优参数：**
```
learning_rate=0.001, dropout=0.3, hidden_dim=128
```

**超参数搜索（12组）：**

| learning_rate | dropout | hidden_dim | val_acc |
|--------------|---------|-----------|---------|
| 0.001 | 0.2 | 128 | 74.35% |
| 0.001 | 0.2 | 256 | 73.97% |
| 0.001 | 0.3 | **128** | **74.90%** ✓ |
| 0.001 | 0.3 | 256 | 74.09% |
| 0.001 | 0.4 | 128 | 74.81% |
| 0.001 | 0.4 | 256 | 74.80% |
| 0.005 | 0.2 | 128 | 74.07% |
| 0.005 | 0.2 | 256 | 74.07% |
| 0.005 | 0.3 | 128 | 74.27% |
| 0.005 | 0.3 | 256 | 74.62% |
| 0.005 | 0.4 | 128 | 74.23% |
| 0.005 | 0.4 | 256 | 74.21% |

**训练损失：**
Epoch 5→30：0.4438 → 0.4086

**分类报告：**

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|----|---------|
| CT | 0.72 | 0.79 | 0.75 | 11799 |
| T  | 0.78 | 0.72 | 0.75 | 12658 |
| **accuracy** | | | **0.75** | 24457 |

---

## 汇总对比

| 模型 | Test Accuracy | Best CV Accuracy |
|------|--------------|-----------------|
| XGBoost | 75.42% | 75.00% |
| Logistic Regression | 74.89% | 74.72% |
| Random Forest | 76.06% | 75.02% |
| KNN | 73.97% | 73.07% |
| Neural Network | 75.08% | 74.90% |