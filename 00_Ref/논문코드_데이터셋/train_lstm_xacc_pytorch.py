"""
train_lstm_xacc_pytorch.py

End-to-end PyTorch LSTM training script for Xacc prediction
using 6-DOF fishing vessel simulation data.

Data source:
    D:\\MSS_6\\fishingVessel\\6Dof_dataset_withXacc\\6Dof_..._withXacc.csv

Features:
    - Time series window (lookback_window seconds) of:
        u, v, w, p, q, r, phi, theta, Xacc  (9 features)

Targets (multi-step forecast):
    - Xacc at future steps:
        Xacc(t+1), Xacc(t+2), ..., Xacc(t+prediction_horizon)

You can control:
    - lookback_window       : length of input window (in seconds)
    - prediction_horizon    : how many seconds ahead to predict
        e.g.
          prediction_horizon = 5  ->  1~5 s ahead (0~5 s 구간 예측)
          prediction_horizon = 10 ->  1~10 s ahead

This script:
    1) Loads all *_withXacc.csv files
    2) Builds supervised sequences
    3) Normalizes features/targets
    4) Trains LSTM (PyTorch)
    5) Evaluates with RMSE / MAE / R^2 per horizon step
    6) Saves model and scalers

Note:
    - All plot labels/titles are in English (per project rule).
"""

import os
import glob
import random
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Configuration
# =========================

# DATA_DIR:
#   - Original mixed dataset : D:\MSS_6\fishingVessel\6Dof_dataset_withXacc
#   - Tonnage-only dataset   : D:\MSS_6\fishingVessel\6Dof_dataset_withXacc_ton
DATA_DIR = r"D:\MSS_6\fishingVessel\6Dof_dataset_withXacc_ton"
OUTPUT_DIR = r"D:\MSS_6\fishingVessel\LSTM_PyTorch"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Time-series configuration
LOOKBACK_WINDOW = 20          # seconds of history (input window length)
# We focus on 3-step (1~3 s ahead) Xacc prediction for visualization
PREDICTION_HORIZON = 3

# If you want 1~10 s ahead later, set PREDICTION_HORIZON = 10.

# Training configuration
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
PATIENCE = 12  # early stopping patience (epochs)

RANDOM_SEED = 42


# =========================
# Reproducibility
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =========================
# Dataset Preparation
# =========================

# Use full 6-DOF velocities, angles, and Xacc as inputs
FEATURE_COLUMNS = ["u", "v", "w", "p", "q", "r", "phi", "theta", "Xacc"]
TARGET_COLUMN = "Xacc"


def load_all_csv_files(data_dir: str) -> List[pd.DataFrame]:
    pattern = os.path.join(data_dir, "6Dof_*_withXacc.csv")
    file_list = sorted(glob.glob(pattern))

    if not file_list:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    print("=" * 70)
    print("Loading CSV files")
    print("=" * 70)
    print(f"Found {len(file_list)} files")

    dataframes = []
    for i, filepath in enumerate(file_list, 1):
        print(f"  [{i}/{len(file_list)}] {os.path.basename(filepath)}")
        df = pd.read_csv(filepath)

        # Basic column check
        missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in file: {filepath}")

        dataframes.append(df)

    print()
    return dataframes


def build_sequences_from_df(
    df: pd.DataFrame,
    lookback_window: int,
    prediction_horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build supervised sequences from a single DataFrame.

    Inputs:
        X(t-lookback+1 ... t)  ->  shape: (lookback_window, num_features)
    Targets (multi-step):
        [Xacc(t+1), ..., Xacc(t+prediction_horizon)]  -> shape: (prediction_horizon,)
    """
    values = df[FEATURE_COLUMNS].values  # (N, num_features)
    xacc = df[TARGET_COLUMN].values      # (N,)

    num_samples = len(df)
    num_features = values.shape[1]

    sequences = []
    targets = []

    # index t is "current" (last index in the input window)
    # we need t+prediction_horizon <= num_samples-1  ->  t <= num_samples-1-prediction_horizon
    start_t = lookback_window - 1
    end_t = num_samples - prediction_horizon - 1

    if end_t <= start_t:
        # Not enough data in this file
        return np.empty((0, lookback_window, num_features)), np.empty((0, prediction_horizon))

    for t in range(start_t, end_t + 1):
        past_start = t - lookback_window + 1
        past_end = t + 1  # exclusive

        # Input window
        window = values[past_start:past_end, :]  # (lookback_window, num_features)

        # Future targets
        future = xacc[t + 1: t + 1 + prediction_horizon]  # (prediction_horizon,)

        if future.shape[0] != prediction_horizon:
            # safety check (should not happen with correct indexing)
            continue

        sequences.append(window)
        targets.append(future)

    sequences = np.stack(sequences, axis=0)  # (num_sequences, lookback_window, num_features)
    targets = np.stack(targets, axis=0)      # (num_sequences, prediction_horizon)

    return sequences, targets


def build_dataset(
    dataframes: List[pd.DataFrame],
    lookback_window: int,
    prediction_horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    all_X = []
    all_y = []

    print("=" * 70)
    print("Building supervised sequences")
    print("=" * 70)

    total_sequences = 0
    for i, df in enumerate(dataframes, 1):
        seq, tgt = build_sequences_from_df(df, lookback_window, prediction_horizon)
        print(
            f"  [{i}/{len(dataframes)}] "
            f"samples in file: {len(df):5d}, sequences created: {len(seq):5d}"
        )
        if len(seq) > 0:
            all_X.append(seq)
            all_y.append(tgt)
            total_sequences += len(seq)

    if not all_X:
        raise RuntimeError("No sequences could be created. Check lookback_window and prediction_horizon.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print()
    print(f"Total sequences: {total_sequences}")
    print(f"Input shape:  {X.shape}  (N, lookback_window, num_features)")
    print(f"Target shape: {y.shape}  (N, prediction_horizon)")
    print()

    return X, y


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    num_samples = X.shape[0]
    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    n_test = num_samples - n_train - n_val

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    print("=" * 70)
    print("Train / Validation / Test split")
    print("=" * 70)
    print(f"Train:       {X_train.shape[0]} samples ({train_ratio * 100:.1f}%)")
    print(f"Validation:  {X_val.shape[0]} samples ({val_ratio * 100:.1f}%)")
    print(f"Test:        {X_test.shape[0]} samples ({test_ratio * 100:.1f}%)")
    print()

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """
    Normalize input features and targets using StandardScaler.
    - Inputs: scale per-feature over all timesteps.
    - Targets: scale separately per horizon step.
    """
    num_features = X_train.shape[2]
    pred_horizon = y_train.shape[1]

    # Flatten over time dimension for feature scaling
    X_train_flat = X_train.reshape(-1, num_features)
    X_val_flat = X_val.reshape(-1, num_features)
    X_test_flat = X_test.reshape(-1, num_features)

    scaler_X = StandardScaler()
    X_train_flat_scaled = scaler_X.fit_transform(X_train_flat)
    X_val_flat_scaled = scaler_X.transform(X_val_flat)
    X_test_flat_scaled = scaler_X.transform(X_test_flat)

    X_train_scaled = X_train_flat_scaled.reshape(X_train.shape)
    X_val_scaled = X_val_flat_scaled.reshape(X_val.shape)
    X_test_scaled = X_test_flat_scaled.reshape(X_test.shape)

    # Targets: scale each horizon step separately
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    print("=" * 70)
    print("Normalization")
    print("=" * 70)
    print(f"Feature scaler: StandardScaler (mean/std per feature)")
    print(f"Target scaler:  StandardScaler (mean/std per horizon step)")
    print(f"X_train mean (scaled): {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
    print(f"y_train mean (scaled): {y_train_scaled.mean():.4f}, std: {y_train_scaled.std():.4f}")
    print()

    return (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y,
    )


class XaccDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, lookback_window, num_features)
        y: (N, prediction_horizon)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# =========================
# Model Definition
# =========================

class XaccLSTM(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        prediction_horizon: int,
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
        self.fc2 = nn.Linear(hidden_size, prediction_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, num_features)
        returns: (batch_size, prediction_horizon)
        """
        out, (h_n, c_n) = self.lstm(x)  # out: (batch, seq_len, hidden)
        last = out[:, -1, :]           # (batch, hidden)
        x = self.dropout(last)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =========================
# Training & Evaluation
# =========================


class WeightedMSELoss(nn.Module):
    """
    Mean Squared Error with higher weight on the last prediction step (e.g., t+5s).
    """

    def __init__(self, last_step_weight: float = 2.0):
        super().__init__()
        self.last_step_weight = last_step_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred, y_true: (batch, prediction_horizon)
        weights = torch.ones_like(y_true)
        # emphasize the last horizon step
        weights[:, -1] = self.last_step_weight
        loss = weights * (y_pred - y_true) ** 2
        return loss.mean()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    running_loss = 0.0
    num_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    return running_loss / max(num_samples, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
) -> float:
    model.eval()
    running_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            batch_size = X_batch.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

    return running_loss / max(num_samples, 1)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    Compute overall MSE, RMSE, MAE, R^2 and per-step RMSE / R^2.
    """
    assert y_true.shape == y_pred.shape
    diff = y_pred - y_true
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    # Step-wise RMSE and R^2
    step_rmse = np.sqrt(np.mean(diff ** 2, axis=0))  # (prediction_horizon,)
    step_r2_list = []
    for k in range(y_true.shape[1]):
        yk = y_true[:, k]
        ypk = y_pred[:, k]
        ss_res_k = np.sum((yk - ypk) ** 2)
        ss_tot_k = np.sum((yk - np.mean(yk)) ** 2)
        r2_k = 1.0 - ss_res_k / (ss_tot_k + 1e-12)
        step_r2_list.append(r2_k)
    step_r2 = np.array(step_r2_list)

    return mse, rmse, mae, r2, step_rmse, step_r2


def main():
    # 1) Load data
    dataframes = load_all_csv_files(DATA_DIR)

    # 2) Build sequences
    X, y = build_dataset(
        dataframes,
        lookback_window=LOOKBACK_WINDOW,
        prediction_horizon=PREDICTION_HORIZON,
    )

    num_samples, seq_len, num_features = X.shape

    # 3) Split
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    # 4) Normalize
    (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y,
    ) = normalize_data(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )

    # 5) Create datasets and loaders
    train_dataset = XaccDataset(X_train_scaled, y_train_scaled)
    val_dataset = XaccDataset(X_val_scaled, y_val_scaled)
    test_dataset = XaccDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 6) Model, optimizer, loss
    model = XaccLSTM(
        num_features=num_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        prediction_horizon=PREDICTION_HORIZON,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Weighted MSE: give more importance to the last prediction step (t+5s)
    criterion = WeightedMSELoss(last_step_weight=1.5)

    print("=" * 70)
    print("Model")
    print("=" * 70)
    print(model)
    print()

    # 7) Training loop with early stopping
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    print("=" * 70)
    print("Training")
    print("=" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:3d} / {NUM_EPOCHS:3d}  "
            f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}"
        )

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 8) Evaluation on test set
    print()
    print("=" * 70)
    print("Evaluation on Test Set")
    print("=" * 70)

    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            y_pred = model(X_batch)

            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y_batch.cpu().numpy())

    y_pred_scaled = np.concatenate(all_preds, axis=0)  # (N_test, PREDICTION_HORIZON)
    y_true_scaled = np.concatenate(all_trues, axis=0)

    # Inverse transform
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_true_scaled)

    mse, rmse, mae, r2, step_rmse, step_r2 = compute_metrics(y_true, y_pred)

    print(f"MSE:  {mse:.8f}")
    print(f"RMSE: {rmse:.6f} rad ({np.degrees(rmse):.4f} deg)")
    print(f"MAE:  {mae:.6f} rad ({np.degrees(mae):.4f} deg)")
    print(f"R^2:  {r2:.4f}")
    print()

    print("Step-wise metrics (future horizon):")
    for i, (s_rmse, s_r2) in enumerate(zip(step_rmse, step_r2), start=1):
        print(
            f"  t+{i:2d} s: RMSE={s_rmse:.6f} rad ({np.degrees(s_rmse):.4f} deg), "
            f"R^2={s_r2:.4f}"
        )
    print()

    # 9) Visualization (English labels)
    print("Creating plots...")

    # 9-1) Training history
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Training History")
    ax1.legend()
    ax1.grid(True)
    fig1.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "training_history.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    # 9-2) Predicted vs Actual for first N samples at last horizon step
    n_plot = min(200, y_true.shape[0])
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    last_step = PREDICTION_HORIZON - 1
    ax2.plot(y_true[:n_plot, last_step], label=f"Actual Xacc(t+{PREDICTION_HORIZON}s)", linewidth=1.5)
    ax2.plot(y_pred[:n_plot, last_step], label=f"Predicted Xacc(t+{PREDICTION_HORIZON}s)", linewidth=1.5)
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Xacc (rad)")
    ax2.set_title(f"Predictions vs Actual (last horizon step: t+{PREDICTION_HORIZON}s)")
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "pred_vs_actual_last_step.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    # 9-2-b) Predicted vs Actual for 3-step horizons (t+1, t+2, t+3)
    #        Visualize each horizon as a time series.
    if PREDICTION_HORIZON >= 3:
        n_plot_multi = min(200, y_true.shape[0])
        fig_multi, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        horizons_to_plot = [1, 2, 3 ]
        for idx, h in enumerate(horizons_to_plot):
            ax = axes[idx]
            step_idx = h - 1
            ax.plot(
                y_true[:n_plot_multi, step_idx],
                label=f"Actual Xacc(t+{h+1}s)",
                linewidth=1.2,
            )
            ax.plot(
                y_pred[:n_plot_multi, step_idx],
                label=f"Predicted Xacc(t+{h+1}s)",
                linewidth=1.2,
            )
            ax.set_ylabel("Xacc (rad)")
            ax.set_title(f"Xacc prediction at t+{h+1}s")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Sample index")
        fig_multi.tight_layout()
        fig_multi_path = os.path.join(OUTPUT_DIR, "pred_vs_actual_3steps_timeseries.png")
        fig_multi.savefig(fig_multi_path, dpi=300)
        plt.close(fig_multi)

    # 9-3) Step-wise RMSE plot
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    horizons = np.arange(1, PREDICTION_HORIZON + 1)
    ax3.plot(horizons, np.degrees(step_rmse), marker="o")
    ax3.set_xlabel("Prediction horizon (s)")
    ax3.set_ylabel("RMSE (deg)")
    ax3.set_title("Step-wise RMSE")
    ax3.grid(True)
    fig3.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, "stepwise_rmse.png")
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)

    print(f"  Training history plot: {fig1_path}")
    print(f"  Prediction plot (last step): {fig2_path}")
    if PREDICTION_HORIZON >= 3:
        print(f"  3-step prediction time-series plot: {fig_multi_path}")
    print(f"  Step-wise RMSE plot: {fig3_path}")
    print()

    # 10) Save model and scalers
    model_path = os.path.join(OUTPUT_DIR, "xacc_lstm_pytorch.pt")
    torch.save(model.state_dict(), model_path)

    # Save scalers as .npz (simple NumPy dict)
    scaler_file = os.path.join(OUTPUT_DIR, "xacc_lstm_scalers.npz")
    np.savez(
        scaler_file,
        scaler_X_mean=scaler_X.mean_,
        scaler_X_scale=scaler_X.scale_,
        scaler_y_mean=scaler_y.mean_,
        scaler_y_scale=scaler_y.scale_,
        feature_columns=np.array(FEATURE_COLUMNS, dtype=object),
    )

    print("Saved:")
    print(f"  Model weights: {model_path}")
    print(f"  Scalers:       {scaler_file}")
    print()
    print("=" * 70)
    print("Done (PyTorch LSTM Xacc prediction)")
    print("=" * 70)


if __name__ == "__main__":
    main()


