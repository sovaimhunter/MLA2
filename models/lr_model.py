from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from utils import load_data


def run():
    X_train, X_test, y_train, y_test = load_data()

    # ── 超参数调优 ──────────────────────────────────────────
    # C        : 正则化强度的倒数，越小正则化越强（搜索范围：0.01 ~ 10）
    # solver   : 优化算法，lbfgs 适合小特征，saga 支持 L1 正则
    param_grid = {
        "C":      [0.01, 0.1, 1.0, 10.0],   # 超参数：正则化系数
        "solver": ["lbfgs", "saga"],          # 超参数：优化器
    }
    # ────────────────────────────────────────────────────────

    base = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    search = GridSearchCV(base, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")

    preds = search.best_estimator_.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["CT", "T"]))