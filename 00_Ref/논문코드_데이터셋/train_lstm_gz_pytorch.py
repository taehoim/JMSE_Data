"""
train_lstm_gz_pytorch.py

Experiment 3: Direct prediction of GZ(t+1..t+5) using LSTM.

Inputs:
  - Past 20 s window of 9 features:
      [u, v, w, p, q, r, phi, theta, Xacc]

Targets:
  - GZ at t+1..t+5 (5-step vector)
"""

import os
import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# Configuration
# =========================

DATA_DIR = r"D:\MSS_6\fishingVessel\6Dof_dataset_withXacc_ton"
OUTPUT_DIR = r"D:\MSS_6\fishingVessel\LSTM_PyTorch"

os.makedirs(OUTPUT_DIR, exist_ok=True)

LOOKBACK_WINDOW = 20
# 3-step (t+1, t+2, t+3) GZ prediction for visualization
PREDICTION_HORIZON = 3

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
PATIENCE = 12

FEATURE_COLUMNS = ["u", "v", "w", "p", "q", "r", "phi", "theta", "Xacc"]
TARGET_COLUMN_GZ = "GZ"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class LSTMGZ(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        out_dim: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        x = self.dropout(last)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_all_csv_files(data_dir: str) -> List[pd.DataFrame]:
    pattern = os.path.join(data_dir, "6Dof_*_withXacc.csv")
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    print("=" * 70)
    print("Loading CSV files for GZ experiment")
    print("=" * 70)
    print(f"Found {len(file_list)} files")

    dfs = []
    for i, fp in enumerate(file_list, 1):
        print(f"  [{i}/{len(file_list)}] {os.path.basename(fp)}")
        df = pd.read_csv(fp)
        missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN_GZ] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {fp}")
        dfs.append(df)
    print()
    return dfs


def build_sequences_gz(
    dfs: List[pd.DataFrame],
    lookback_window: int,
    prediction_horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    all_X = []
    all_y = []

    print("=" * 70)
    print("Building sequences for GZ")
    print("=" * 70)

    total_sequences = 0

    for file_idx, df in enumerate(dfs, 1):
        print(f"  File {file_idx}/{len(dfs)}: {len(df)} samples")
        values = df[FEATURE_COLUMNS].values
        gz = df[TARGET_COLUMN_GZ].values

        num_samples = len(df)
        num_features = values.shape[1]

        start_t = lookback_window - 1
        end_t = num_samples - prediction_horizon - 1
        if end_t <= start_t:
            continue

        for t in range(start_t, end_t + 1):
            past_start = t - lookback_window + 1
            past_end = t + 1
            window = values[past_start:past_end, :]

            gz_future = gz[t + 1 : t + 1 + prediction_horizon]
            if gz_future.shape[0] != prediction_horizon:
                continue

            all_X.append(window)
            all_y.append(gz_future)
            total_sequences += 1

    X = np.stack(all_X, axis=0)
    y = np.stack(all_y, axis=0)

    print(f"Total sequences: {total_sequences}")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print()

    return X, y


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    n = X.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]

    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
):
    num_features = X_train.shape[2]

    X_train_flat = X_train.reshape(-1, num_features)
    X_val_flat = X_val.reshape(-1, num_features)
    X_test_flat = X_test.reshape(-1, num_features)

    scaler_X = StandardScaler()
    X_train_flat_s = scaler_X.fit_transform(X_train_flat)
    X_val_flat_s = scaler_X.transform(X_val_flat)
    X_test_flat_s = scaler_X.transform(X_test_flat)

    X_train_s = X_train_flat_s.reshape(X_train.shape)
    X_val_s = X_val_flat_s.reshape(X_val.shape)
    X_test_s = X_test_flat_s.reshape(X_test.shape)

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)

    return X_train_s, y_train_s, X_val_s, X_test_s, scaler_X, scaler_y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    assert y_true.shape == y_pred.shape
    diff = y_pred - y_true
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    step_rmse = np.sqrt(np.mean(diff ** 2, axis=0))
    step_r2 = []
    for k in range(y_true.shape[1]):
        yk = y_true[:, k]
        ypk = y_pred[:, k]
        ss_res_k = np.sum((yk - ypk) ** 2)
        ss_tot_k = np.sum((yk - np.mean(yk)) ** 2)
        r2_k = 1.0 - ss_res_k / (ss_tot_k + 1e-12)
        step_r2.append(r2_k)
    step_r2 = np.array(step_r2)
    return mse, rmse, mae, r2, step_rmse, step_r2


def main():
    dfs = load_all_csv_files(DATA_DIR)
    X, y = build_sequences_gz(
        dfs, lookback_window=LOOKBACK_WINDOW, prediction_horizon=PREDICTION_HORIZON
    )

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    X_train_s, y_train_s, X_val_s, X_test_s, scaler_X, scaler_y = normalize_data(
        X_train, y_train, X_val, X_test
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train_s, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),  # unscaled for monitoring
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_s, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_features = X.shape[2]

    model = LSTMGZ(
        num_features=num_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        out_dim=PREDICTION_HORIZON,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Standard MSE loss (no weighting)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    print("=" * 70)
    print("Training LSTM for direct GZ prediction")
    print("=" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running = 0.0
        n_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            optimizer.step()
            bs = xb.size(0)
            running += loss.item() * bs
            n_samples += bs
        train_loss = running / max(n_samples, 1)

        model.eval()
        running = 0.0
        n_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb_s = torch.tensor(
                    scaler_y.transform(yb.cpu().numpy()), dtype=torch.float32
                ).to(DEVICE)
                yhat = model(xb)
                loss = criterion(yhat, yb_s)
                bs = xb.size(0)
                running += loss.item() * bs
                n_samples += bs
        val_loss = running / max(n_samples, 1)

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS:3d}  "
            f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    preds_s = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yhat = model(xb)
            preds_s.append(yhat.cpu().numpy())

    y_pred_s = np.concatenate(preds_s, axis=0)
    y_pred = scaler_y.inverse_transform(y_pred_s)
    y_true = y_test

    mse, rmse, mae, r2, step_rmse, step_r2 = compute_metrics(y_true, y_pred)

    print("=" * 70)
    print("GZ prediction metrics")
    print("=" * 70)
    print(f"MSE:  {mse:.8f} m^2")
    print(f"RMSE: {rmse:.6f} m")
    print(f"MAE:  {mae:.6f} m")
    print(f"R^2:  {r2:.4f}")
    print("Step-wise metrics (m):")
    for i, (s_rmse, s_r2) in enumerate(zip(step_rmse, step_r2), 1):
        print(f"  t+{i:2d}s: RMSE={s_rmse:.6f} m, R^2={s_r2:.4f}")
    print()

    # Visualization: 3-step GZ predictions vs actual (time series)
    if PREDICTION_HORIZON >= 3:
        print("Creating 3-step GZ prediction plots...")
        n_plot = min(200, y_true.shape[0])
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        horizons_to_plot = [1, 2, 3]  # t+1s, t+2s, t+3s
        for idx, h in enumerate(horizons_to_plot):
            ax = axes[idx]
            step_idx = h - 1
            ax.plot(
                y_true[:n_plot, step_idx],
                label=f"Actual GZ(t+{h+1}s)",
                linewidth=1.2,
            )
            ax.plot(
                y_pred[:n_plot, step_idx],
                label=f"Predicted GZ(t+{h+1}s)",
                linewidth=1.2,
            )
            ax.set_ylabel("GZ (m)")
            ax.set_title(f"GZ prediction at t+{h+1}s")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Sample index")
        fig.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, "gz_pred_vs_actual_3steps_timeseries.png")
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"  3-step GZ prediction time-series plot: {fig_path}")
        print()

    model_path = os.path.join(OUTPUT_DIR, "gz_lstm.pt")
    torch.save(model.state_dict(), model_path)
    scaler_path = os.path.join(OUTPUT_DIR, "gz_lstm_scalers.npz")
    np.savez(
        scaler_path,
        scaler_X_mean=scaler_X.mean_,
        scaler_X_scale=scaler_X.scale_,
        scaler_y_mean=scaler_y.mean_,
        scaler_y_scale=scaler_y.scale_,
        feature_columns=np.array(FEATURE_COLUMNS, dtype=object),
    )

    print(f"Model saved to:   {model_path}")
    print(f"Scalers saved to: {scaler_path}")
    print()


if __name__ == "__main__":
    main()


