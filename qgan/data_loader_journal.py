# qgan/data_loader_journal.py
# Multi-subject data loader with Leave-One-Out Cross-Validation (LOOCV)
# for journal extension.
#
# DROP-IN: import from this file instead of data_loader.py for journal runs.
#
# Subjects:  3 (EPCTL01, EPCTL02, EPCTL03) — add more paths to SUBJECT_PATHS
# LOOCV:     train on N-1 subjects, test on 1, repeat N times
# Output:    mean ± std across folds reported in paper
#
# Usage:
#   from qgan.data_loader_journal import load_all_subjects, loocv_split, SUBJECT_PATHS
#   all_data = load_all_subjects(n_features=4)
#   train_data, test_data = loocv_split(all_data, test_idx=0)

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyedflib

from qgan.config import EEG_CHANNEL, EPOCH_SECONDS, BATCH_SIZE

# ================================================================
#  SUBJECT FILE PATHS
#  Add more subjects here when you download them.
#  Journal target: 7+ subjects for LOOCV generalisation claim.
# ================================================================
SUBJECT_PATHS = [
    "data/EPCTL01.edf",
    "data/EPCTL02.edf",
    "data/EPCTL03.edf",
    "data/EPCTL04.edf",   # add when available
    "data/EPCTL05.edf",
    "data/EPCTL06.edf",
    "data/EPCTL07.edf",
    "data/EPCTL09.edf",
    "data/EPCTL10.edf",
    "data/EPCTL11.edf",
]

# Cache: path -> tensor, so LOOCV folds don't re-read disk
_SUBJECT_CACHE = {}


# ================================================================
#  LOAD ONE SUBJECT
# ================================================================
def load_subject(path: str, n_features: int = 4) -> torch.Tensor:
    """
    Load one EDF file and return normalised feature tensor [n_epochs, n_features].
    Features: [Mean, Std Dev, Min, Max] — first n_features used.
    Normalised to [-1, 1] per feature (matches quantum circuit output range).
    Cached after first load.
    """
    cache_key = (path, n_features)
    if cache_key in _SUBJECT_CACHE:
        return _SUBJECT_CACHE[cache_key]

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"EDF file not found: {path}\n"
            f"Make sure the file is in the data/ folder."
        )

    with pyedflib.EdfReader(path) as f:
        signal      = f.readSignal(EEG_CHANNEL)
        sample_rate = f.getSampleFrequency(EEG_CHANNEL)

    samples_per_epoch = int(EPOCH_SECONDS * sample_rate)
    n_epochs          = len(signal) // samples_per_epoch

    if n_epochs < 50:
        raise ValueError(
            f"Subject {path} has only {n_epochs} epochs — too short. "
            f"Need at least 50 epochs."
        )

    signal = signal[: n_epochs * samples_per_epoch]
    epochs = signal.reshape(n_epochs, samples_per_epoch)

    # Extract features
    all_feats = np.array(
        [[e.mean(), e.std(), e.min(), e.max()] for e in epochs],
        dtype=np.float32
    )
    feats = all_feats[:, :n_features]

    # Normalise each feature column to [-1, 1]
    for col in range(n_features):
        lo, hi = feats[:, col].min(), feats[:, col].max()
        if hi - lo > 1e-8:
            feats[:, col] = 2.0 * (feats[:, col] - lo) / (hi - lo) - 1.0
        else:
            feats[:, col] = 0.0

    tensor = torch.from_numpy(feats)
    _SUBJECT_CACHE[cache_key] = tensor
    return tensor


# ================================================================
#  LOAD ALL SUBJECTS
# ================================================================
def load_all_subjects(n_features: int = 4,
                      paths: list = None) -> list:
    """
    Load all subjects. Returns list of tensors, one per subject.
    paths defaults to SUBJECT_PATHS if not provided.

    Returns:
        all_data: list of tensors, each [n_epochs_i, n_features]
    """
    if paths is None:
        paths = SUBJECT_PATHS

    all_data = []
    print(f"\n  Loading {len(paths)} subjects | {n_features} features")
    print(f"  {'─'*55}")

    for path in paths:
        tensor = load_subject(path, n_features)
        all_data.append(tensor)
        print(f"    {os.path.basename(path):<25} {tensor.shape[0]} epochs")

    total = sum(t.shape[0] for t in all_data)
    print(f"  {'─'*55}")
    print(f"  Total: {total} epochs across {len(all_data)} subjects\n")
    return all_data


# ================================================================
#  LOOCV SPLIT
# ================================================================
def loocv_split(all_data: list, test_idx: int):
    """
    Leave-One-Out split.
    Train: all subjects except test_idx (concatenated)
    Test:  subject at test_idx only

    Args:
        all_data : list of tensors [n_subjects], each [n_epochs_i, n_features]
        test_idx : index of the held-out test subject (0 to n_subjects-1)

    Returns:
        train_tensor : [sum of train epochs, n_features]
        test_tensor  : [test subject epochs, n_features]
    """
    train = torch.cat([d for i, d in enumerate(all_data) if i != test_idx])
    test  = all_data[test_idx]
    return train, test


# ================================================================
#  GET DATA LOADER — LOOCV AWARE
# ================================================================
def get_loocv_loader(all_data: list,
                     test_idx: int,
                     batch_size: int = BATCH_SIZE,
                     noise_level: float = 0.0):
    """
    Build DataLoader for one LOOCV fold.

    Args:
        all_data    : output of load_all_subjects()
        test_idx    : which subject is held out for testing
        batch_size  : training batch size
        noise_level : if > 0, add Gaussian noise to training data
                      (use 0.1 for Simulator+DataNoise condition)

    Returns:
        train_loader : DataLoader for training
        test_data    : tensor for evaluation [n_test_epochs, n_features]
    """
    train_data, test_data = loocv_split(all_data, test_idx)

    if noise_level > 0.0:
        noise      = torch.randn_like(train_data) * noise_level
        train_data = torch.clamp(train_data + noise, -1.0, 1.0)

    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return train_loader, test_data


# ================================================================
#  SUBJECT CHECKER — run standalone to verify all files
#  python -c "from qgan.data_loader_journal import check_subjects; check_subjects()"
# ================================================================
def check_subjects(paths: list = None):
    """
    Verify all EDF files load correctly. Run this before starting experiments.
    Prints a table showing each file's status, epoch count, and feature stats.
    """
    if paths is None:
        paths = SUBJECT_PATHS

    print(f"\n  {'='*65}")
    print(f"  SUBJECT CHECK")
    print(f"  {'─'*65}")
    print(f"  {'File':<28} {'Epochs':<10} {'Features':<12} Status")
    print(f"  {'─'*65}")

    valid = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  {os.path.basename(path):<28} {'---':<10} {'---':<12} "
                  f"FILE NOT FOUND")
            continue
        try:
            t = load_subject(path, n_features=4)
            n_epochs = t.shape[0]
            status   = "OK" if n_epochs >= 50 else "TOO SHORT (<50 epochs)"
            print(f"  {os.path.basename(path):<28} {n_epochs:<10} "
                  f"{'4':<12} {status}")
            if n_epochs >= 50:
                valid.append(path)
        except Exception as e:
            print(f"  {os.path.basename(path):<28} {'---':<10} {'---':<12} "
                  f"ERROR: {e}")

    print(f"  {'─'*65}")
    print(f"  Valid subjects: {len(valid)} / {len(paths)}")
    print(f"  LOOCV folds possible: {len(valid)}")

    total = 0
    for path in valid:
        t = load_subject(path, n_features=4)
        total += t.shape[0]
    print(f"  Total epochs (valid): {total}")
    print(f"  {'='*65}\n")
    return valid


if __name__ == "__main__":
    check_subjects()