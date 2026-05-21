"""
CS:GO Round Winner Classifier
Usage:
  python run_models.py rf
  python run_models.py xgboost
  python run_models.py lr
  python run_models.py knn
  python run_models.py nn
  python run_models.py all
"""

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = "csgo_round_snapshots.csv"
_cache = {}


# ── Data loading ────────────────────────────────────────────

def _make_round_ids(df):
    return (df["time_left"].diff().fillna(0) > 0).cumsum()


def load_data(test_size=0.2, scale=True):
    key = (test_size, scale)
    if key in _cache:
        return _cache[key]

    df = pd.read_csv(DATA_PATH)
    df["map"] = LabelEncoder().fit_transform(df["map"])
    df["bomb_planted"] = df["bomb_planted"].astype(int)

    round_ids = _make_round_ids(df)
    print(f"Total rounds: {round_ids.nunique()}")

    X = df.drop(columns=["round_winner"]).values.astype(np.float32)
    y = LabelEncoder().fit_transform(df["round_winner"])  # CT=0, T=1

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=round_ids))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    _cache[key] = (X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


# ── Models ───────────────────────────────────────────────────

def run_rf():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = load_data()

    param_dist = {
        "n_estimators":      [300, 500, 800, 1000],
        "max_depth":         [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", 0.3, 0.5, 0.7],
    }

    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=30, cv=3,
        scoring="accuracy", random_state=42, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    preds = search.best_estimator_.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["CT", "T"]))


def run_xgboost():
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = load_data()

    # lr=0.01 需要 1000+ 棵树才能收敛，此处改用较大学习率避免欠拟合
    param_dist = {
        "max_depth":        [4, 5, 6],
        "learning_rate":    [0.05, 0.1],
        "n_estimators":     [300, 500, 700],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_alpha":        [0, 0.1, 0.5],
    }

    base = xgb.XGBClassifier(
        eval_metric="logloss", tree_method="hist",
        random_state=42, n_jobs=-1
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=30, cv=3,
        scoring="accuracy", random_state=42, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    preds = search.best_estimator_.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["CT", "T"]))


def run_lr():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = load_data()

    param_grid = {
        "C":      [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "saga"],
    }

    base = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    search = GridSearchCV(base, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    preds = search.best_estimator_.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["CT", "T"]))


def run_knn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = load_data()

    param_grid = {
        "n_neighbors": [5, 7, 11, 15, 21],
        "metric":      ["euclidean", "manhattan"],
    }

    base = KNeighborsClassifier(n_jobs=-1)
    search = GridSearchCV(base, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    preds = search.best_estimator_.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["CT", "T"]))


def run_nn():
    import itertools
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = load_data()

    SEARCH_SPACE = {
        "learning_rate": [1e-3, 5e-3],
        "dropout":       [0.2, 0.3, 0.4],
        "hidden_dim":    [128, 256],
    }
    VAL_RATIO = 0.15
    EPOCHS_SEARCH = 10
    EPOCHS_FINAL = 30

    class RoundNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, dropout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x)

    def _train_eval(X_tr, y_tr, X_val, y_val, lr, dropout, hidden_dim, epochs, device):
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=512, shuffle=True)
        model = RoundNet(X_tr.shape[1], hidden_dim, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_val_t).squeeze(1).cpu().numpy()
        return accuracy_score(y_val, (logits >= 0).astype(int)), model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_val = int(len(X_train) * VAL_RATIO)
    X_tr, X_val = X_train[n_val:], X_train[:n_val]
    y_tr, y_val = y_train[n_val:], y_train[:n_val]

    keys = list(SEARCH_SPACE.keys())
    combos = list(itertools.product(*SEARCH_SPACE.values()))
    print(f"\nSearching {len(combos)} hyperparameter combinations...\n")

    best_val_acc, best_params = 0, {}
    for combo in combos:
        params = dict(zip(keys, combo))
        val_acc, _ = _train_eval(
            X_tr, y_tr, X_val, y_val,
            lr=params["learning_rate"], dropout=params["dropout"],
            hidden_dim=params["hidden_dim"], epochs=EPOCHS_SEARCH, device=device,
        )
        print(f"  {params}  →  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc, best_params = val_acc, params

    print(f"\nBest params: {best_params}")
    print(f"Best CV accuracy: {best_val_acc:.4f}")
    print(f"\nRetraining with best params for {EPOCHS_FINAL} epochs...")

    X_tr_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=512, shuffle=True)
    model = RoundNet(X_train.shape[1], best_params["hidden_dim"], best_params["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"], weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(1, EPOCHS_FINAL + 1):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}/{EPOCHS_FINAL}  loss={total_loss / len(X_train):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_te_t).squeeze(1).cpu().numpy()
    preds = (logits >= 0).astype(int)
    print(f"\nTest Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["CT", "T"]))


# ── Entry point ──────────────────────────────────────────────

RUNNERS = {
    "rf":      run_rf,
    "xgboost": run_xgboost,
    "lr":      run_lr,
    "knn":     run_knn,
    "nn":      run_nn,
}


def main():
    parser = argparse.ArgumentParser(description="CS:GO Round Winner Classifier")
    parser.add_argument("model", choices=list(RUNNERS.keys()) + ["all"],
                        help="Model to train, or 'all' to run every model")
    args = parser.parse_args()

    targets = list(RUNNERS.keys()) if args.model == "all" else [args.model]
    for name in targets:
        print(f"\n{'='*40}\n  Training: {name.upper()}\n{'='*40}\n")
        RUNNERS[name]()


if __name__ == "__main__":
    main()
