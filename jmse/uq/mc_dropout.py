"""MC-Dropout uncertainty: T stochastic forward passes with dropout left active.

At inference we re-enable dropout (and recurrent inter-layer dropout) so each pass
samples a different sub-network; the spread across T passes is the epistemic-ish
predictive uncertainty. Our models contain no BatchNorm, so toggling the whole module
to train() is safe; `enable_dropout` is the surgical alternative that flips only the
Dropout layers (recurrent dropout inside nn.LSTM/GRU is then left off).
"""
import numpy as np
import torch
from torch import nn


def enable_dropout(model: nn.Module) -> None:
    """Put only nn.Dropout* layers into train() mode (leave everything else in eval())."""
    for m in model.modules():
        if isinstance(m, nn.modules.dropout._DropoutNd):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, X, T: int = 30, device="cpu", target_scaler=None,
                       batch_size: int = 1024):
    """Return (mean, std) over T dropout-active passes, shapes (N, horizon), in target units.

    Sets the model to train() so all dropout (head + recurrent) is sampled, runs T
    passes, inverse-scales each, and reduces over the pass axis. The model's original
    train/eval mode is restored before returning.
    """
    was_training = model.training
    model.train()                                    # activate dropout (no BatchNorm here)
    model.to(device)
    X = torch.as_tensor(np.asarray(X, dtype=np.float32))

    passes = []
    for _ in range(T):
        outs = []
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(device)
            outs.append(model(xb).cpu().numpy())
        y = np.concatenate(outs, axis=0) if outs else np.empty((0,))
        if target_scaler is not None:
            y = target_scaler.inverse_transform(y)
        passes.append(y)

    if not was_training:
        model.eval()
    arr = np.stack(passes, axis=0)                   # (T, N, horizon)
    return arr.mean(axis=0), arr.std(axis=0)
