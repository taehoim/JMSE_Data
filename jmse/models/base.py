"""Base forecaster: shared inference helper and seeding."""
import os
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int = 0):
    """Seed all RNGs and request deterministic cuDNN (for reproducible GPU sweeps)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BaseForecaster(nn.Module):
    """nn.Module subclass with a batched numpy predict() that inverse-scales targets."""

    @torch.no_grad()
    def predict(self, X, device="cpu", batch_size=1024, target_scaler=None) -> np.ndarray:
        self.eval()
        self.to(device)
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        outs = []
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(device)
            outs.append(self(xb).cpu().numpy())
        y = np.concatenate(outs, axis=0) if outs else np.empty((0,))
        if target_scaler is not None:
            y = target_scaler.inverse_transform(y)
        return y
