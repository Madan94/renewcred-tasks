from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class SoCLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.lstm1(x)
        o = self.dropout(o)
        o, _ = self.lstm2(o)
        o = o[:, -1, :]
        return self.fc(o).squeeze(-1)


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    device_ids: np.ndarray,
    seq_len: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-device sequences: row i uses timesteps [i-seq_len+1 : i+1], target y[i].
    Only when all seq_len rows belong to the same device.
    Returns (seq_X, y, seq_device_id) for each sequence (device of last timestep).
    """
    seqs: List[np.ndarray] = []
    targets: List[float] = []
    devs: List[str] = []
    n = len(y)
    start = 0
    while start < n:
        end = start
        while end + 1 < n and device_ids[end + 1] == device_ids[start]:
            end += 1
        for i in range(start + seq_len - 1, end + 1):
            if device_ids[i] != device_ids[i - seq_len + 1]:
                continue
            seqs.append(X[i - seq_len + 1 : i + 1])
            targets.append(float(y[i]))
            devs.append(str(device_ids[i]))
        start = end + 1
    if not seqs:
        return (
            np.empty((0, seq_len, X.shape[1])),
            np.array([], dtype=np.float32),
            np.array([], dtype=object),
        )
    return (
        np.stack(seqs, axis=0),
        np.array(targets, dtype=np.float32),
        np.array(devs, dtype=object),
    )


def train_lstm_soc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dev_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dev_test: np.ndarray,
    *,
    plot_dir: Path | None = None,
    seq_len: int = 10,
    epochs: int = 80,
    batch_size: int = 256,
    patience: int = 10,
    lr: float = 1e-3,
    device: str | None = None,
) -> Tuple[SoCLSTM, StandardScaler, Dict[str, Any]]:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train.shape[1]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    Xtr_seq, ytr_seq, dev_tr_seq = build_sequences(X_tr_s, y_train, dev_train, seq_len=seq_len)
    Xte_seq, yte_seq, dev_te_seq = build_sequences(X_te_s, y_test, dev_test, seq_len=seq_len)

    if len(Xtr_seq) < 10 or len(Xte_seq) < 1:
        raise RuntimeError("Not enough sequence rows for LSTM; check data size per device.")

    # validation = last 15% of training sequences (time-ordered within concatenated order)
    n_val = max(1, int(0.15 * len(Xtr_seq)))
    n_val = min(n_val, max(0, len(Xtr_seq) - 1))
    if n_val < 1 or len(Xtr_seq) - n_val < 1:
        n_val = max(1, len(Xtr_seq) // 10)
    Xv, yv = Xtr_seq[-n_val:], ytr_seq[-n_val:]
    Xtr, ytr = Xtr_seq[:-n_val], ytr_seq[:-n_val]
    if len(Xtr) < 1:
        Xtr, ytr = Xtr_seq, ytr_seq
        Xv, yv = Xtr_seq[-n_val:], ytr_seq[-n_val:]

    model = SoCLSTM(n_features=n_features).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tr_ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).float())
    va_ds = TensorDataset(torch.from_numpy(Xv).float(), torch.from_numpy(yv).float())
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    best_loss = float("inf")
    best_state = None
    stale = 0
    epoch = -1
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            losses = []
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb)
                losses.append(loss_fn(pred, yb).item())
        va_loss = float(np.mean(losses)) if losses else best_loss
        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    train_time_sec = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        te_t = torch.from_numpy(Xte_seq).float().to(dev)
        pred_te = model(te_t).cpu().numpy()
    rmse = float(np.sqrt(mean_squared_error(yte_seq, pred_te)))
    mae = float(mean_absolute_error(yte_seq, pred_te))

    if plot_dir is not None and len(dev_te_seq) > 0:
        plot_dir.mkdir(parents=True, exist_ok=True)
        top_devs = pd.Series(dev_te_seq).value_counts().head(3).index.tolist()
        for d in top_devs:
            m = dev_te_seq == d
            if m.sum() < 5:
                continue
            plot_pred_vs_actual(
                yte_seq[m],
                pred_te[m],
                f"LSTM pred vs actual (device {str(d)[:12]})",
                plot_dir / f"lstm_pred_vs_actual_{str(d)[:10]}.png",
            )

    # latency
    t_inf0 = time.perf_counter()
    with torch.no_grad():
        _ = model(torch.from_numpy(Xte_seq[: min(500, len(Xte_seq))]).float().to(dev))
    t_inf1 = time.perf_counter()
    n_inf = min(500, len(Xte_seq))
    latency_ms = (t_inf1 - t_inf0) / max(n_inf, 1) * 1000.0

    metrics = {
        "test_rmse": rmse,
        "test_mae": mae,
        "training_time_sec": train_time_sec,
        "inference_latency_ms_per_sample": latency_ms,
        "epochs_ran": int(epoch + 1),
    }
    return model, scaler, metrics


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    lim = max(y_true.max(), y_pred.max(), 100)
    plt.scatter(y_true, y_pred, s=8, alpha=0.5)
    plt.plot([0, lim], [0, lim], "k--", lw=1)
    plt.xlabel("Actual SoC (%)")
    plt.ylabel("Predicted SoC (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_lstm_bundle(
    model: SoCLSTM,
    scaler: StandardScaler,
    n_features: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler": scaler,
            "n_features": n_features,
        },
        path,
    )


def metrics_to_json(metrics: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()}, f, indent=2)
