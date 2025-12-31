import time
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

from .config import (
    MAX_EPOCHS,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
)


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = BATCH_SIZE,
):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_optimizer(model: nn.Module, optimizer_name: str, lr: float = LEARNING_RATE):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'")


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in data_loader:
            probs = model(xb)
            preds = probs.argmax(dim=1)
            all_probs.append(probs[:, 1].cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    loss = log_loss(y_true, np.vstack([1 - y_prob, y_prob]).T, labels=[0, 1])

    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "loss": loss,
    }


def train_qnn(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer_name: str,
) -> Tuple[Dict[str, float], Dict[str, float], float, int]:
    """Train model with early stopping."""
    criterion = nn.NLLLoss()

    def loss_fn(probs, targets):
        log_probs = torch.log(probs + 1e-9)
        return criterion(log_probs, targets)

    optimizer = get_optimizer(model, optimizer_name)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    start_time = time.time()
    epochs_run = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            probs = model(xb)
            loss = loss_fn(probs, yb)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_model(model, val_loader)
        val_loss = val_metrics["loss"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        epochs_run = epoch + 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break

    train_time = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate_model(model, train_loader)
    val_metrics = evaluate_model(model, val_loader)

    return val_metrics, train_metrics, train_time, epochs_run
