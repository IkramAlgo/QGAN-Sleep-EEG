# qgan/data_loader.py
# Loads Sleep EDF and returns first N features normalized to [-1, 1]

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyedflib

from qgan.config import EDF_FILE_PATH, EEG_CHANNEL, EPOCH_SECONDS, ALL_FEATURE_NAMES, BATCH_SIZE

# cache so we don't re-read the EDF file for every experiment
_cache = None


def load_sleep_edf(path=EDF_FILE_PATH):
    global _cache
    if _cache is not None:
        return _cache

    with pyedflib.EdfReader(path) as f:
        signal      = f.readSignal(EEG_CHANNEL)
        sample_rate = f.getSampleFrequency(EEG_CHANNEL)

    samples_per_epoch = int(EPOCH_SECONDS * sample_rate)
    n_epochs          = len(signal) // samples_per_epoch

    if n_epochs == 0:
        raise ValueError("EDF file too short for a single 30-second epoch.")

    signal = signal[: n_epochs * samples_per_epoch]
    epochs = signal.reshape(n_epochs, samples_per_epoch)

    # extract all 4 features — experiments pick first N
    features = np.array(
        [[e.mean(), e.std(), e.min(), e.max()] for e in epochs],
        dtype=np.float32
    )

    # normalize each column to [-1, 1]
    for i in range(features.shape[1]):
        lo, hi = features[:, i].min(), features[:, i].max()
        features[:, i] = (2.0 * (features[:, i] - lo) / (hi - lo) - 1.0) \
                         if hi - lo > 1e-8 else 0.0

    _cache = torch.from_numpy(features)
    return _cache


def get_data_loader(n_features, batch_size=BATCH_SIZE, path=EDF_FILE_PATH):
    """Return DataLoader using only the first n_features columns."""
    data = load_sleep_edf(path)[:, :n_features]
    print(f"  Dataset: {data.shape[0]} epochs | {n_features} features "
          f"({ALL_FEATURE_NAMES[:n_features]}) | range [-1,1]")
    return DataLoader(TensorDataset(data), batch_size=batch_size,
                      shuffle=True, drop_last=True)