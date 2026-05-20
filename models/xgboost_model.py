import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from utils import load_data


def run():
    X_train, X_test, y_train, y_test = load_data()

    # ── 超参数调优 ──────────────────────────────────────────
    # max_depth      : 树的最大深度，越深越容易过拟合（搜索范围：3 ~ 9）
    # learning_rate  : 每棵树的收缩步长，越小需要越多树（搜索范围：0.01 ~ 0.3）
    # n_estimators   : 树的总数量（搜索范围：100 ~ 500）
    # subsample      : 每棵树随机采样的样本比例（搜索范围：0.6 ~ 1.0）
    # colsample_bytree: 每棵树随机采样的特征比例（搜索范围：0.6 ~ 1.0）
    param_dist = {
        "max_depth":       [3, 5, 6, 9],          # 超参数：树的最大深度
        "learning_rate":   [0.01, 0.05, 0.1, 0.3],# 超参数：学习率
        "n_estimators":    [100, 200, 300, 500],   # 超参数：树的数量
        "subsample":       [0.6, 0.8, 1.0],        # 超参数：样本采样比例
        "colsample_bytree":[0.6, 0.8, 1.0],        # 超参数：特征采样比例
    }
    # ────────────────────────────────────────────────────────

    base = xgb.XGBClassifier(
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=15, cv=3,
        scoring="accuracy", random_state=42, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")

    preds = search.best_estimator_.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["CT", "T"]))
