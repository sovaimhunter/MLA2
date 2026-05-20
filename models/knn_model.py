from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from utils import load_data


def run():
    X_train, X_test, y_train, y_test = load_data()

    # ── 超参数调优 ──────────────────────────────────────────
    # n_neighbors: 近邻数量，越小越容易过拟合（搜索范围：5 ~ 21）
    # metric     : 距离度量方式，影响邻域定义（euclidean / manhattan）
    param_grid = {
        "n_neighbors": [5, 7, 11, 15, 21],              # 超参数：近邻数量 k
        "metric":      ["euclidean", "manhattan"],        # 超参数：距离度量
    }
    # ────────────────────────────────────────────────────────

    base = KNeighborsClassifier(n_jobs=-1)
    search = GridSearchCV(base, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")

    preds = search.best_estimator_.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["CT", "T"]))
