from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from utils import load_data


def run():
    X_train, X_test, y_train, y_test = load_data()

    # ── 超参数调优 ──────────────────────────────────────────
    # n_estimators     : 树的数量，越多越稳定但更慢（搜索范围：100 ~ 500）
    # max_depth        : 单棵树最大深度，None 表示不限制（搜索范围：10 ~ None）
    # min_samples_split: 节点分裂所需最小样本数，越大越保守（搜索范围：2 ~ 10）
    param_dist = {
        "n_estimators":      [100, 200, 300, 500],   # 超参数：树的数量
        "max_depth":         [10, 20, 30, None],      # 超参数：树的最大深度
        "min_samples_split": [2, 5, 10],              # 超参数：最小分裂样本数
    }
    # ────────────────────────────────────────────────────────

    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=12, cv=3,
        scoring="accuracy", random_state=42, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")

    preds = search.best_estimator_.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["CT", "T"]))
