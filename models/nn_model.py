import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from utils import load_data
import itertools


# ── 超参数调优 ──────────────────────────────────────────
# learning_rate: Adam 优化器步长（搜索范围：1e-3 ~ 1e-2）
# dropout      : Dropout 比例，防止过拟合（搜索范围：0.2 ~ 0.4）
# hidden_dim   : 第一隐藏层宽度，决定模型容量（搜索范围：128 ~ 256）
SEARCH_SPACE = {
    "learning_rate": [1e-3, 5e-3],   # 超参数：学习率
    "dropout":       [0.2, 0.3, 0.4],# 超参数：Dropout 比例
    "hidden_dim":    [128, 256],      # 超参数：隐藏层维度
}
# ────────────────────────────────────────────────────────

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


def run():
    X_train, X_test, y_train, y_test = load_data()

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
            lr=params["learning_rate"],
            dropout=params["dropout"],
            hidden_dim=params["hidden_dim"],
            epochs=EPOCHS_SEARCH,
            device=device,
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
