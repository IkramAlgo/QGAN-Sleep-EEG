# qgan/data_loader.py
# Updated with normalization for stable quantum circuit training

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyedflib


def load_sleep_edf(path="data/EPCTL03.edf"):
    """Return normalized array shape (n_epochs, n_features).

    Features per 30-second epoch:
        0 - mean
        1 - std
        2 - min
        3 - max

    All features are normalized to [-1, 1] range which matches
    the output range of quantum PauliZ expectation values.
    """
    with pyedflib.EdfReader(path) as f:
        sigs = f.readSignal(0)        # first EEG channel
        fs   = f.getSampleFrequency(0)

    samples_per_epoch = int(30 * fs)
    n_epochs = len(sigs) // samples_per_epoch
    if n_epochs == 0:
        raise ValueError("EDF recording too short for a single 30-sec epoch")

    trimmed = sigs[: n_epochs * samples_per_epoch]
    epochs  = trimmed.reshape(n_epochs, samples_per_epoch)

    feat_list = []
    for e in epochs:
        mean = float(e.mean())
        std  = float(e.std())
        mn   = float(e.min())
        mx   = float(e.max())
        feat_list.append([mean, std, mn, mx])

    feats = np.array(feat_list, dtype=np.float32)

    # Normalize each feature to [-1, 1] range
    # This is critical for quantum circuits which output in [-1, 1]
    for i in range(feats.shape[1]):
        col = feats[:, i]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 1e-8:
            feats[:, i] = 2.0 * (col - col_min) / (col_max - col_min) - 1.0
        else:
            feats[:, i] = 0.0

    print(f"Loaded {n_epochs} sleep epochs | Feature shape: {feats.shape}")
    print(f"Feature ranges after normalization:")
    feature_names = ["mean", "std", "min", "max"]
    for i, name in enumerate(feature_names):
        print(f"  {name}: [{feats[:,i].min():.3f}, {feats[:,i].max():.3f}]")

    return torch.from_numpy(feats)


def get_data_loader(batch_size):
    data = load_sleep_edf()
    return DataLoader(
        TensorDataset(data),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True    # avoid incomplete final batch issues
    )