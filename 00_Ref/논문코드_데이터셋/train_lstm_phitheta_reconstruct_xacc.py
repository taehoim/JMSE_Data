"""
train_lstm_phitheta_reconstruct_xacc.py

Experiment 2 (phi/theta): 
  - Predict future phi, theta, then reconstruct Xacc = sqrt(phi^2 + theta^2)
    and compare with true Xacc.

Inputs:
  - Past 20 s window of 9 features:
      [u, v, w, p, q, r, phi, theta, Xacc]

Targets:
  - phi(t+1..t+5), theta(t+1..t+5)  → 10-dimensional output.
"""

import os
import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
PREDICTION_HORIZON = 5

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
TARGET_COLS = ["phi", "theta"]
TARGET_XACC_COL = "Xacc"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# =========================
# Model
# =========================

class LSTMphiTheta(nn.Module):
    def __init__(self, num_features: int, hidden_size: int, num_layers: int, dropout: float, out_dim: int):
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


# =========================
# Data utilities
# =========================

def load_all_csv_files(data_dir: str) -> List[pd.DataFrame]:
    pattern = os.path.join(data_dir, "6Dof_*_withXacc.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No *_withXacc.csv in {data_dir}")

    print("=" * 70)
    print("Loading CSV files for phi/theta experiment")
    print("=" * 70)
    print(f"Found {len(files)} files")

    dfs = []
    for i, fp in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {os.path.basename(fp)}")
        df = pd.read_csv(fp)
        missing = [c for c in FEATURE_COLUMNS + [TARGET_XACC_COL] + TARGET_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing {missing} in {fp}")
        dfs.append(df)
    print()
    return dfs


def build_sequences(
    dfs: List[pd.DataFrame],
    lookback_window: int,
    prediction_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: (N, 20, 9)
      y_phitheta: (N, 10)  [phi1..phi5, theta1..theta5]
      y_xacc: (N, 5)  true Xacc(t+1..t+5)
    """
    all_X = []
    all_y_pt = []
    all_y_xacc = []

    print("=" * 70)
    print("Building sequences for phi/theta")
    print("=" * 70)

    total = 0

    for file_idx, df in enumerate(dfs, 1):
        print(f"  File {file_idx}/{len(dfs)}: {len(df)} samples")
        values = df[FEATURE_COLUMNS].values
        xacc = df[TARGET_XACC_COL].values
        phi = df["phi"].values
        theta = df["theta"].values

        n = len(df)
        num_features = values.shape[1]

        start_t = lookback_window - 1
        end_t = n - prediction_horizon - 1
        if end_t <= start_t:
            continue

        for t in range(start_t, end_t + 1):
            past_start = t - lookback_window + 1
            past_end = t + 1

            Xwin = values[past_start:past_end, :]  # (20,9)

            # future phi/theta
            phi_f = phi[t+1:t+1+prediction_horizon]
            theta_f = theta[t+1:t+1+prediction_horizon]
            if len(phi_f) != prediction_horizon or len(theta_f) != prediction_horizon:
                continue
            y_pt = np.concatenate([phi_f, theta_f], axis=0)  # (10,)

            # future Xacc (true)
            xacc_f = xacc[t+1:t+1+prediction_horizon]
            if len(xacc_f) != prediction_horizon:
                continue

            all_X.append(Xwin)
            all_y_pt.append(y_pt)
            all_y_xacc.append(xacc_f)
            total += 1

    X = np.stack(all_X, axis=0)
    y_pt = np.stack(all_y_pt, axis=0)
    y_xacc = np.stack(all_y_xacc, axis=0)

    print(f"Total sequences: {total}")
    print("Input shape:", X.shape)
    print("phi/theta target shape:", y_pt.shape)
    print("Xacc target shape:", y_xacc.shape)
    print()

    return X, y_pt, y_xacc


def split_data(
    X: np.ndarray,
    y_pt: np.ndarray,
    y_xacc: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    n = X.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train = X[:n_train]
    y_pt_train = y_pt[:n_train]
    y_xacc_train = y_xacc[:n_train]

    X_val = X[n_train:n_train+n_val]
    y_pt_val = y_pt[n_train:n_train+n_val]
    y_xacc_val = y_xacc[n_train:n_train+n_val]

    X_test = X[n_train+n_val:]
    y_pt_test = y_pt[n_train+n_val:]
    y_xacc_test = y_xacc[n_train+n_val:]

    return X_train, y_pt_train, y_xacc_train, X_val, y_pt_val, y_xacc_val, X_test, y_pt_test, y_xacc_test


def normalize_X_y(
    X_train, y_train, X_val, X_test
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
    """
    Compute overall MSE, RMSE, MAE, R^2 and per-step RMSE/R^2 (if y is 2D).
    """
    diff = y_pred - y_true
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - ss_res/(ss_tot + 1e-12)

    # if y is 2D (N, H or N, num_components), compute per-dim RMSE/R^2
    if y_true.ndim == 2:
        step_rmse = np.sqrt(np.mean(diff**2, axis=0))
        step_r2_list = []
        for k in range(y_true.shape[1]):
            yk = y_true[:, k]
            ypk = y_pred[:, k]
            ss_res_k = np.sum((yk - ypk) ** 2)
            ss_tot_k = np.sum((yk - np.mean(yk)) ** 2)
            r2_k = 1.0 - ss_res_k / (ss_tot_k + 1e-12)
            step_r2_list.append(r2_k)
        step_r2 = np.array(step_r2_list)
    else:
        step_rmse = np.array([])
        step_r2 = np.array([])

    return mse, rmse, mae, r2, step_rmse, step_r2


def main():
    dfs = load_all_csv_files(DATA_DIR)
    X, y_pt, y_xacc = build_sequences(dfs, LOOKBACK_WINDOW, PREDICTION_HORIZON)

    (
        X_train,
        y_pt_train,
        y_xacc_train,
        X_val,
        y_pt_val,
        y_xacc_val,
        X_test,
        y_pt_test,
        y_xacc_test,
    ) = split_data(X, y_pt, y_xacc, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    X_train_s, y_train_s, X_val_s, X_test_s, scaler_X, scaler_y = normalize_X_y(
        X_train, y_pt_train, X_val, X_test
    )

    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train_s, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(y_pt_val, dtype=torch.float32),  # unscaled for val
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_s, dtype=torch.float32),
        torch.tensor(y_pt_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMphiTheta(
        num_features=X.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        out_dim=y_pt.shape[1],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0

    print("="*70)
    print("Training LSTM for phi/theta prediction")
    print("="*70)

    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        run = 0.0
        n_s = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            optimizer.step()
            bs = xb.size(0)
            run += loss.item()*bs
            n_s += bs
        train_loss = run/max(n_s,1)

        # validation (scale targets on the fly)
        model.eval()
        run = 0.0
        n_s = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb_s = torch.tensor(
                    scaler_y.transform(yb.cpu().numpy()),
                    dtype=torch.float32,
                ).to(DEVICE)
                yhat = model(xb)
                loss = criterion(yhat, yb_s)
                bs = xb.size(0)
                run += loss.item()*bs
                n_s += bs
        val_loss = run/max(n_s,1)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS:3d}  Train: {train_loss:.6f}  Val: {val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping at epoch", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test: predict phi/theta → reconstruct Xacc
    model.eval()
    preds_s = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yhat = model(xb)
            preds_s.append(yhat.cpu().numpy())

    y_pred_s = np.concatenate(preds_s, axis=0)
    y_pred_flat = scaler_y.inverse_transform(y_pred_s)

    # reshape to (N, H, 2)
    y_pred_pt = y_pred_flat.reshape(-1, PREDICTION_HORIZON, 2)
    y_true_pt = y_pt_test.reshape(-1, PREDICTION_HORIZON, 2)

    # component-level metrics
    # Overall (phi+theta 함께 평탄화)
    mse_c, rmse_c, mae_c, r2_c, _, _ = compute_metrics(
        y_true_pt.reshape(-1, 2), y_pred_pt.reshape(-1, 2)
    )

    # phi, theta 각각에 대해 step별 RMSE/R^2 계산
    mse_phi, rmse_phi, mae_phi, r2_phi, step_rmse_phi, step_r2_phi = compute_metrics(
        y_true_pt[:, :, 0], y_pred_pt[:, :, 0]
    )
    mse_theta, rmse_theta, mae_theta, r2_theta, step_rmse_theta, step_r2_theta = compute_metrics(
        y_true_pt[:, :, 1], y_pred_pt[:, :, 1]
    )

    print("="*70)
    print("phi/theta prediction metrics")
    print("="*70)
    print(f"Overall (phi+theta flattened):")
    print(f"  MSE : {mse_c:.8f}")
    print(f"  RMSE: {rmse_c:.6f}")
    print(f"  MAE : {mae_c:.6f}")
    print(f"  R^2 : {r2_c:.4f}")
    print()

    print("phi(t+1..t+5):")
    print(f"  RMSE (overall): {rmse_phi:.6f} rad ({np.degrees(rmse_phi):.4f} deg)")
    print(f"  MAE  (overall): {mae_phi:.6f} rad ({np.degrees(mae_phi):.4f} deg)")
    print(f"  R^2  (overall): {r2_phi:.4f}")
    print("  Step-wise (deg, R^2):")
    for i, (s_rmse, s_r2) in enumerate(zip(step_rmse_phi, step_r2_phi), 1):
        print(f"    t+{i:2d}s: RMSE={np.degrees(s_rmse):.4f} deg, R^2={s_r2:.4f}")
    print()

    print("theta(t+1..t+5):")
    print(f"  RMSE (overall): {rmse_theta:.6f} rad ({np.degrees(rmse_theta):.4f} deg)")
    print(f"  MAE  (overall): {mae_theta:.6f} rad ({np.degrees(mae_theta):.4f} deg)")
    print(f"  R^2  (overall): {r2_theta:.4f}")
    print("  Step-wise (deg, R^2):")
    for i, (s_rmse, s_r2) in enumerate(zip(step_rmse_theta, step_r2_theta), 1):
        print(f"    t+{i:2d}s: RMSE={np.degrees(s_rmse):.4f} deg, R^2={s_r2:.4f}")
    print()

    # Reconstruct Xacc
    phi_pred = y_pred_pt[:, :, 0]
    theta_pred = y_pred_pt[:, :, 1]
    xacc_pred = np.sqrt(phi_pred**2 + theta_pred**2)

    xacc_true = y_xacc_test  # (N,H)

    mse_x, rmse_x, mae_x, r2_x, step_rmse_x, step_r2_x = compute_metrics(xacc_true, xacc_pred)
    print("="*70)
    print("Reconstructed Xacc metrics from phi/theta")
    print("="*70)
    print(f"MSE:  {mse_x:.8f}")
    print(f"RMSE: {rmse_x:.6f} rad ({np.degrees(rmse_x):.4f} deg)")
    print(f"MAE:  {mae_x:.6f} rad ({np.degrees(mae_x):.4f} deg)")
    print(f"R^2:  {r2_x:.4f}")
    print("Step-wise metrics (deg):")
    for i, (s_rmse, s_r2) in enumerate(zip(step_rmse_x, step_r2_x), 1):
        print(f"  t+{i:2d}s: RMSE={np.degrees(s_rmse):.4f} deg, R^2={s_r2:.4f}")
    print()

    # Save model/scalers
    model_path = os.path.join(OUTPUT_DIR, "phitheta_lstm.pt")
    torch.save(model.state_dict(), model_path)
    scaler_path = os.path.join(OUTPUT_DIR, "phitheta_lstm_scalers.npz")
    np.savez(
        scaler_path,
        scaler_X_mean=scaler_X.mean_,
        scaler_X_scale=scaler_X.scale_,
        scaler_y_mean=scaler_y.mean_,
        scaler_y_scale=scaler_y.scale_,
        feature_columns=np.array(FEATURE_COLUMNS, dtype=object),
        target_cols=np.array(TARGET_COLS, dtype=object),
    )
    print("Model saved to:", model_path)
    print("Scalers saved to:", scaler_path)


if __name__ == "__main__":
    main()