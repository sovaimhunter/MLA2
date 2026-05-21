"""
RQ1: How do different ML model categories compare when predicting CS:GO round winners?
Usage:
  python rq1_models.py rf
  python rq1_models.py xgboost
  python rq1_models.py lr
  python rq1_models.py nn
  python rq1_models.py all
"""

import argparse
import itertools
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from shared import load_data


def _get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def run_rf():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    print("\n" + "="*55)
    print("  RQ1 — Random Forest")
    print("="*55)

    X_train, X_test, y_train, y_test = load_data()

    base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    base.fit(X_train, y_train)
    preds_base = base.predict(X_test)
    acc_before = accuracy_score(y_test, preds_base)
    f1_before  = f1_score(y_test, preds_base, average="weighted")
    print(f"Baseline  acc={acc_before:.4f}  f1={f1_before:.4f}")

    param_dist = {
        "n_estimators":      [300, 500, 800, 1000],
        "max_depth":         [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", 0.3, 0.5, 0.7],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_dist, n_iter=30, cv=3,
        scoring="accuracy", random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    preds_tuned = search.best_estimator_.predict(X_test)
    acc_after = accuracy_score(y_test, preds_tuned)
    f1_after  = f1_score(y_test, preds_tuned, average="weighted")

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Tuned     acc={acc_after:.4f}  f1={f1_after:.4f}")
    print(classification_report(y_test, preds_tuned, target_names=["CT", "T"]))

    return {"model": "Random Forest", "acc_before": acc_before, "acc_after": acc_after,
            "f1_before": f1_before, "f1_after": f1_after}


def run_xgboost():
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV

    print("\n" + "="*55)
    print("  RQ1 — XGBoost")
    print("="*55)

    X_train, X_test, y_train, y_test = load_data()

    device = _get_device()
    print(f"XGBoost using device: {device}")

    base = xgb.XGBClassifier(
        eval_metric="logloss", tree_method="hist",
        device=device, random_state=42, n_jobs=-1,
    )
    base.fit(X_train, y_train)
    preds_base = base.predict(X_test)
    acc_before = accuracy_score(y_test, preds_base)
    f1_before  = f1_score(y_test, preds_base, average="weighted")
    print(f"Baseline  acc={acc_before:.4f}  f1={f1_before:.4f}")

    param_dist = {
        "max_depth":        [4, 5, 6],
        "learning_rate":    [0.05, 0.1],
        "n_estimators":     [300, 500, 700],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_alpha":        [0, 0.1, 0.5],
    }
    search = RandomizedSearchCV(
        xgb.XGBClassifier(eval_metric="logloss", tree_method="hist",
                           device=device, random_state=42, n_jobs=-1),
        param_dist, n_iter=30, cv=3,
        scoring="accuracy", random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    preds_tuned = search.best_estimator_.predict(X_test)
    acc_after = accuracy_score(y_test, preds_tuned)
    f1_after  = f1_score(y_test, preds_tuned, average="weighted")

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Tuned     acc={acc_after:.4f}  f1={f1_after:.4f}")
    print(classification_report(y_test, preds_tuned, target_names=["CT", "T"]))

    return {"model": "XGBoost", "acc_before": acc_before, "acc_after": acc_after,
            "f1_before": f1_before, "f1_after": f1_after}


def run_lr():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    print("\n" + "="*55)
    print("  RQ1 — Logistic Regression")
    print("="*55)

    X_train, X_test, y_train, y_test = load_data()

    base = LogisticRegression(max_iter=1000, random_state=42)
    base.fit(X_train, y_train)
    preds_base = base.predict(X_test)
    acc_before = accuracy_score(y_test, preds_base)
    f1_before  = f1_score(y_test, preds_base, average="weighted")
    print(f"Baseline  acc={acc_before:.4f}  f1={f1_before:.4f}")

    param_grid = {
        "C":      [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "saga"],
    }
    search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    preds_tuned = search.best_estimator_.predict(X_test)
    acc_after = accuracy_score(y_test, preds_tuned)
    f1_after  = f1_score(y_test, preds_tuned, average="weighted")

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Tuned     acc={acc_after:.4f}  f1={f1_after:.4f}")
    print(classification_report(y_test, preds_tuned, target_names=["CT", "T"]))

    return {"model": "Logistic Regression", "acc_before": acc_before, "acc_after": acc_after,
            "f1_before": f1_before, "f1_after": f1_after}


def run_nn():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import GroupShuffleSplit

    print("\n" + "="*55)
    print("  RQ1 — Neural Network")
    print("="*55)

    X_train, X_test, y_train, y_test, train_round_ids = load_data(return_groups=True)

    SEARCH_SPACE = {
        "learning_rate": [1e-3, 5e-3],
        "dropout":       [0.2, 0.3, 0.4],
        "hidden_dim":    [128, 256],
    }
    VAL_RATIO    = 0.15
    EPOCHS_SRCH  = 50
    EPOCHS_FINAL = 100

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

    def _train(X_tr, y_tr, lr, dropout, hidden_dim, epochs, device):
        X_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=512, shuffle=True)
        model  = RoundNet(X_tr.shape[1], hidden_dim, dropout).to(device)
        opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        crit   = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()
        return model

    def _predict(model, X, device):
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze(1).cpu().numpy()
        return (logits >= 0).astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=42)
    tr_idx, val_idx = next(gss_val.split(X_train, y_train, groups=train_round_ids))
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    # Baseline: default small network
    base_model = _train(X_tr, y_tr, lr=1e-3, dropout=0.3, hidden_dim=128,
                        epochs=EPOCHS_FINAL, device=device)
    preds_base = _predict(base_model, X_test, device)
    acc_before = accuracy_score(y_test, preds_base)
    f1_before  = f1_score(y_test, preds_base, average="weighted")
    print(f"Baseline  acc={acc_before:.4f}  f1={f1_before:.4f}")

    # Hyperparameter search
    keys   = list(SEARCH_SPACE.keys())
    combos = list(itertools.product(*SEARCH_SPACE.values()))
    print(f"\nSearching {len(combos)} hyperparameter combinations...")
    best_val_acc, best_params = 0, {}
    for combo in combos:
        params = dict(zip(keys, combo))
        m = _train(X_tr, y_tr, lr=params["learning_rate"], dropout=params["dropout"],
                   hidden_dim=params["hidden_dim"], epochs=EPOCHS_SRCH, device=device)
        val_acc = accuracy_score(y_val, _predict(m, X_val, device))
        print(f"  {params}  →  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc, best_params = val_acc, params

    print(f"\nBest params: {best_params}  (val_acc={best_val_acc:.4f})")
    print(f"Retraining for {EPOCHS_FINAL} epochs...")

    # Retrain on full training set with scheduler
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=512, shuffle=True)
    model  = RoundNet(X_train.shape[1], best_params["hidden_dim"], best_params["dropout"]).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"], weight_decay=1e-4)
    crit   = nn.BCEWithLogitsLoss()
    sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    for epoch in range(1, EPOCHS_FINAL + 1):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        sched.step()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}/{EPOCHS_FINAL}  loss={total_loss / len(X_train):.4f}")

    preds_tuned = _predict(model, X_test, device)
    acc_after = accuracy_score(y_test, preds_tuned)
    f1_after  = f1_score(y_test, preds_tuned, average="weighted")

    print(f"\nTuned     acc={acc_after:.4f}  f1={f1_after:.4f}")
    print(classification_report(y_test, preds_tuned, target_names=["CT", "T"]))

    return {"model": "Neural Network", "acc_before": acc_before, "acc_after": acc_after,
            "f1_before": f1_before, "f1_after": f1_after}


RUNNERS = {
    "rf":      run_rf,
    "xgboost": run_xgboost,
    "lr":      run_lr,
    "nn":      run_nn,
}


def run_all():
    results = [fn() for fn in RUNNERS.values()]

    print("\n\n" + "="*68)
    print("  RQ1 Summary: Model Comparison (Baseline vs Tuned)")
    print("="*68)
    print(f"  {'Model':<22} {'Acc Base':>9} {'Acc Tuned':>10} {'F1 Base':>9} {'F1 Tuned':>9}")
    print("  " + "-"*62)
    for r in results:
        print(f"  {r['model']:<22} {r['acc_before']:>9.4f} {r['acc_after']:>10.4f} "
              f"{r['f1_before']:>9.4f} {r['f1_after']:>9.4f}")


def main():
    parser = argparse.ArgumentParser(description="RQ1 — CS:GO Round Winner: Model Comparison")
    parser.add_argument("model", choices=list(RUNNERS.keys()) + ["all"])
    args = parser.parse_args()
    if args.model == "all":
        run_all()
    else:
        RUNNERS[args.model]()


if __name__ == "__main__":
    main()
