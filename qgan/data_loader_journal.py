# qgan/data_loader_journal.py
# Multi-subject data loader with LOOCV for journal extension.
#
# ── JOURNAL UPGRADES vs CONFERENCE ─────────────────────────────────────────
#   1. SPECTRAL FEATURES — delta/theta/alpha/sigma/beta band power via Welch PSD
#      ANPHY-Sleep at 1000 Hz resolves all AASM bands reliably.
#   2. COMBINED FEATURE SETS — statistical (4) + spectral (5) = 9 features
#   3. STAGE LABEL LOADING — returns stage labels for downstream evaluation
#   4. MINORITY OVERSAMPLING — applied before GAN training
#   5. AUGMENTED DATASET BUILDER — for downstream SVM/RF evaluation
#
# ── BUG FIXES vs PREVIOUS JOURNAL VERSION ──────────────────────────────────
#   FIX 1 (CRITICAL): Normalization is now done GLOBALLY across all subjects
#          after loading, not per-subject independently.  Per-subject [-1,1]
#          normalization means the same float value represents different absolute
#          EEG magnitudes in different subjects, breaking cross-subject LOOCV.
#          Solution: compute column min/max over ALL subjects, store the scaler,
#          apply to every subject with the same statistics.
#
#   FIX 2 (CRITICAL): build_augmented_dataset() now detects the minority class
#          dynamically per fold instead of hardcoding target_stage=1 (N1).
#          On some subjects N3 or REM can be rarer than N1.
#
#   FIX 3: Annotation parsing now counts and warns on unmatched annotation
#          strings.  Previously, unrecognised annotations silently defaulted
#          to Wake(0), corrupting stage distributions without any warning.
#
#   FIX 4: get_loocv_loader() now RETURNS train_feats and train_labels as
#          well, so train_journal.py can pass the correct training split to
#          evaluate_downstream() instead of incorrectly passing test data.
#
# ── FEATURE SETS ────────────────────────────────────────────────────────────
#   "statistical"  : [mean, std, min, max]               — 4 features
#   "spectral"     : [delta, theta, alpha, sigma, beta]  — 5 features
#   "combined"     : statistical + spectral               — 9 features
#
# ── LOOCV ───────────────────────────────────────────────────────────────────
#   Train on N-1 subjects, test on 1 held-out, repeat N times.
#
# Usage:
#   from qgan.data_loader_journal import load_all_subjects, get_loocv_loader
#   all_feats, all_labels = load_all_subjects(feature_set="combined")
#   loader, train_f, train_l, test_f, test_l = get_loocv_loader(
#       all_feats, all_labels, test_idx=0)

import os
import warnings
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyedflib

from qgan.config import EEG_CHANNEL, EPOCH_SECONDS, BATCH_SIZE

# ================================================================
#  SUBJECT FILE PATHS
# ================================================================
SUBJECT_PATHS = [
    "data/EPCTL01.edf",
    "data/EPCTL02.edf",
    "data/EPCTL03.edf",
    "data/EPCTL04.edf",
    "data/EPCTL05.edf",
    "data/EPCTL06.edf",
    "data/EPCTL07.edf",
    "data/EPCTL09.edf",
    "data/EPCTL10.edf",
    "data/EPCTL11.edf",
]

# ================================================================
#  FREQUENCY BANDS (Hz) — AASM standard
#  Berry et al. (2012) AASM Scoring Manual v2.4
#  ANPHY-Sleep at 1000 Hz resolves all bands cleanly.
#  Sleep-EDF at 100 Hz is marginal above 30 Hz.
# ================================================================
FREQ_BANDS = {
    "delta": (0.5, 4.0),    # N3 / slow-wave sleep
    "theta": (4.0, 8.0),    # N1 / REM
    "alpha": (8.0, 13.0),   # relaxed wakefulness / N1 onset
    "sigma": (12.0, 15.0),  # sleep spindles — N2 biomarker
    "beta":  (15.0, 30.0),  # active wakefulness / REM
}

# AASM 5-class sleep stage label map
# Keys are checked with `key in annotation_string` (case-insensitive upper)
# Order matters: more specific strings must come before substrings.
# "REM" before "R" prevents "R" matching "REM" incorrectly.
STAGE_MAP_ORDERED = [
    ("STAGE N3",  3),
    ("STAGE N2",  2),
    ("STAGE N1",  1),
    ("STAGE REM", 4),
    ("STAGE R",   4),
    ("STAGE W",   0),
    ("N3",        3),
    ("N2",        2),
    ("N1",        1),
    ("REM",       4),
    (" R ",       4),   # bare "R" with spaces to avoid partial matches
    ("WAKE",      0),
    ("W",         0),
    ("L",        -1),   # lights-on — excluded
]

# Cache: (path, feature_set) -> (raw_feat_matrix unnormalized, label_array)
# Normalization is applied AFTER loading ALL subjects — see load_all_subjects()
_RAW_SUBJECT_CACHE: dict = {}

# ================================================================
#  SPECTRAL BAND POWER
# ================================================================
def _band_power(epoch: np.ndarray, fs: float,
                low: float, high: float) -> float:
    """
    Welch PSD estimate of mean band power in [low, high] Hz.
    nperseg = min(256, n//4) balances freq resolution vs variance.
    Returns 0.0 if the band cannot be resolved.
    """
    try:
        from scipy.signal import welch
    except ImportError:
        raise ImportError("scipy required for spectral features: pip install scipy")

    n = len(epoch)
    nperseg = min(256, n // 4)
    if nperseg < 16:
        return 0.0

    freqs, psd = welch(epoch, fs=fs, window="hann", nperseg=nperseg)
    mask = (freqs >= low) & (freqs < high)
    return float(np.mean(psd[mask])) if np.any(mask) else 0.0


# ================================================================
#  FEATURE EXTRACTION — one epoch
# ================================================================
def _extract_features(epoch: np.ndarray, fs: float,
                       feature_set: str = "statistical") -> np.ndarray:
    """
    Extract features from one 30-s EEG epoch.
    Returns 1D array of length 4 (statistical), 5 (spectral), or 9 (combined).
    """
    feats = []
    if feature_set in ("statistical", "combined"):
        feats += [epoch.mean(), epoch.std(), epoch.min(), epoch.max()]
    if feature_set in ("spectral", "combined"):
        for _, (lo, hi) in FREQ_BANDS.items():
            feats.append(_band_power(epoch, fs, lo, hi))
    if feature_set not in ("statistical", "spectral", "combined"):
        raise ValueError(f"Unknown feature_set='{feature_set}'. "
                         f"Choose: statistical | spectral | combined")
    return np.array(feats, dtype=np.float32)


def _n_features_for_set(feature_set: str) -> int:
    return {"statistical": 4, "spectral": 5, "combined": 9}[feature_set]


# ================================================================
#  LOAD ONE SUBJECT — returns RAW (unnormalized) features + labels
#  FIX 3: annotation warning counter added
# ================================================================
def _load_subject_raw(path: str,
                      feature_set: str = "statistical",
                      channel_idx: int = None) -> tuple:
    """
    Load one EDF file. Returns (raw_feature_matrix, label_array).
    Features are NOT normalized here — normalization happens globally
    in load_all_subjects() after all subjects are loaded.

    Returns:
        feat_matrix : np.ndarray [n_valid_epochs, n_features], unnormalized
        labels      : np.ndarray [n_valid_epochs], int stage labels
                      Lights-on epochs (L=-1) are excluded.
    """
    cache_key = (path, feature_set, channel_idx)
    if cache_key in _RAW_SUBJECT_CACHE:
        return _RAW_SUBJECT_CACHE[cache_key]

    if not os.path.exists(path):
        raise FileNotFoundError(f"EDF not found: {path}")

    chan = channel_idx if channel_idx is not None else EEG_CHANNEL

    with pyedflib.EdfReader(path) as f:
        signal      = f.readSignal(chan)
        sample_rate = f.getSampleFrequency(chan)
        try:
            ann         = f.readAnnotations()
            ann_onsets  = ann[0]
            ann_descs   = ann[2]
            has_ann     = True
        except Exception:
            has_ann = False
            warnings.warn(
                f"No annotations in {path} — all epochs assigned Wake(0). "
                f"Downstream stage labels will be unreliable.",
                stacklevel=2,
            )

    samples_per_epoch = int(EPOCH_SECONDS * sample_rate)
    n_epochs          = len(signal) // samples_per_epoch

    if n_epochs < 50:
        raise ValueError(f"{path}: only {n_epochs} epochs (need >= 50).")

    signal     = signal[:n_epochs * samples_per_epoch]
    epochs_raw = signal.reshape(n_epochs, samples_per_epoch)

    # Build stage label array with unmatched-annotation tracking (FIX 3)
    stage_labels = np.zeros(n_epochs, dtype=np.int64)
    n_unmatched  = 0

    if has_ann:
        for onset, desc in zip(ann_onsets, ann_descs):
            idx = int(onset // EPOCH_SECONDS)
            if not (0 <= idx < n_epochs):
                continue
            desc_up  = str(desc).strip().upper()
            matched  = False
            for key, val in STAGE_MAP_ORDERED:   # ordered — specific before general
                if key in desc_up:
                    stage_labels[idx] = val
                    matched = True
                    break
            if not matched:
                n_unmatched += 1

    if n_unmatched > 0:
        warnings.warn(
            f"{os.path.basename(path)}: {n_unmatched} annotation(s) did not match "
            f"any STAGE_MAP key and defaulted to Wake(0). "
            f"Check annotation strings if stage distribution looks wrong.",
            stacklevel=2,
        )

    # Exclude lights-on (label == -1)
    valid        = stage_labels != -1
    epochs_raw   = epochs_raw[valid]
    stage_labels = stage_labels[valid]
    n_valid      = len(epochs_raw)

    if n_valid < 50:
        raise ValueError(
            f"{path}: only {n_valid} valid epochs after excluding lights-on."
        )

    expected_n  = _n_features_for_set(feature_set)
    feat_matrix = np.zeros((n_valid, expected_n), dtype=np.float32)
    for i, epoch in enumerate(epochs_raw):
        raw = _extract_features(epoch, sample_rate, feature_set)
        feat_matrix[i] = raw[:expected_n]

    # Store RAW — do NOT normalize here
    _RAW_SUBJECT_CACHE[cache_key] = (feat_matrix, stage_labels)
    return feat_matrix, stage_labels


# ================================================================
#  GLOBAL NORMALIZATION — FIX 1
#  Compute column statistics over ALL subjects, then apply uniformly.
#  This ensures the same float value represents the same physical EEG
#  magnitude regardless of which subject it comes from.
# ================================================================
def _normalize_globally(raw_matrices: list) -> tuple:
    """
    Compute column-wise min/max over ALL subjects combined,
    then normalize each subject's matrix to [-1, 1] with the SAME scaler.

    Returns:
        normalized_matrices : list of np.ndarray, same order as input
        scaler              : dict {"min": array, "max": array} for reporting
    """
    # Stack all epochs to find global min/max per feature
    all_data = np.vstack(raw_matrices)                # [total_epochs, n_features]
    col_min  = all_data.min(axis=0)                   # [n_features]
    col_max  = all_data.max(axis=0)                   # [n_features]

    normalized = []
    for mat in raw_matrices:
        norm = np.zeros_like(mat)
        for col in range(mat.shape[1]):
            lo, hi = col_min[col], col_max[col]
            if hi - lo > 1e-8:
                norm[:, col] = 2.0 * (mat[:, col] - lo) / (hi - lo) - 1.0
            else:
                norm[:, col] = 0.0
        normalized.append(norm)

    scaler = {
        "global_min": col_min.tolist(),
        "global_max": col_max.tolist(),
    }
    return normalized, scaler


# ================================================================
#  LOAD ALL SUBJECTS
# ================================================================
def load_all_subjects(n_features: int = 4,
                      feature_set: str = "statistical",
                      paths: list = None) -> tuple:
    """
    Load all subjects, apply GLOBAL normalization (FIX 1).
    Returns (list_of_feat_tensors, list_of_label_tensors, scaler_dict).

    Missing paths are warned and skipped.
    """
    if paths is None:
        paths = SUBJECT_PATHS

    available = [p for p in paths if os.path.exists(p)]
    missing   = [p for p in paths if not os.path.exists(p)]
    if missing:
        warnings.warn(f"Skipping {len(missing)} missing EDF file(s): {missing}")
    if not available:
        raise FileNotFoundError("No EDF files found. Check SUBJECT_PATHS.")

    print(f"\n  {'='*65}")
    print(f"  LOADING SUBJECTS | feature_set={feature_set} | n_features={n_features}")
    print(f"  Subjects: {len(available)} available / {len(paths)} listed")
    print(f"  Normalization: GLOBAL (across all subjects) — FIX 1")
    print(f"  {'─'*65}")

    raw_matrices, label_arrays, names = [], [], []
    for path in available:
        try:
            raw, labs = _load_subject_raw(path, feature_set=feature_set)
            raw_matrices.append(raw[:, :n_features])
            label_arrays.append(labs)
            names.append(os.path.basename(path))
        except Exception as e:
            warnings.warn(f"Skipping {path}: {e}")

    if len(raw_matrices) < 2:
        raise ValueError(
            f"Need >= 2 valid subjects for LOOCV. Got {len(raw_matrices)}."
        )

    # Apply global normalization (FIX 1)
    normalized_matrices, scaler = _normalize_globally(raw_matrices)

    all_features, all_labels = [], []
    for name, norm, labs in zip(names, normalized_matrices, label_arrays):
        feat_t  = torch.from_numpy(norm.astype(np.float32))
        label_t = torch.from_numpy(labs)
        all_features.append(feat_t)
        all_labels.append(label_t)

        stage_counts = {
            s: int((label_t == v).sum())
            for s, v in [("W",0),("N1",1),("N2",2),("N3",3),("REM",4)]
            if int((label_t == v).sum()) > 0
        }
        print(f"    {name:<25} {feat_t.shape[0]:>4} epochs | {stage_counts}")

    total      = sum(t.shape[0] for t in all_features)
    cat_labels = torch.cat(all_labels)
    print(f"  {'─'*65}")
    print(f"  Total: {total} epochs across {len(all_features)} subjects")
    print(f"  Global scaler applied: "
          f"min={[round(v,3) for v in scaler['global_min']]}  "
          f"max={[round(v,3) for v in scaler['global_max']]}")

    print(f"\n  Stage distribution (all subjects combined):")
    for sname, sint in [("W",0),("N1",1),("N2",2),("N3",3),("REM",4)]:
        cnt = int((cat_labels == sint).sum())
        if cnt > 0:
            pct = 100.0 * cnt / len(cat_labels)
            bar = "=" * int(pct / 2)
            print(f"    {sname:<5} {cnt:>5} ({pct:5.1f}%)  {bar}")
    print(f"  {'='*65}\n")

    return all_features, all_labels, scaler


# ================================================================
#  LOOCV SPLIT
# ================================================================
def loocv_split(all_features: list, all_labels: list,
                test_idx: int) -> tuple:
    """
    Leave-One-Out split.
    Returns: train_feats, train_labels, test_feats, test_labels
    """
    train_f = torch.cat([f for i, f in enumerate(all_features) if i != test_idx])
    train_l = torch.cat([l for i, l in enumerate(all_labels)   if i != test_idx])
    return train_f, train_l, all_features[test_idx], all_labels[test_idx]


# ================================================================
#  MINORITY OVERSAMPLING
# ================================================================
def oversample_minority(features: torch.Tensor,
                         labels: torch.Tensor) -> tuple:
    """
    Random oversample each minority class to match the majority class count.
    Applied before GAN training to ensure balanced generator input.
    """
    unique, counts = labels.unique(return_counts=True)
    max_count      = int(counts.max())
    feat_parts     = [features]
    label_parts    = [labels]

    for stage, count in zip(unique, counts):
        shortfall = max_count - int(count)
        if shortfall <= 0:
            continue
        mask = (labels == stage)
        sf   = features[mask]
        idx  = torch.randint(0, int(count), (shortfall,))
        feat_parts.append(sf[idx])
        label_parts.append(labels[mask][idx])

    out_f = torch.cat(feat_parts, dim=0)
    out_l = torch.cat(label_parts, dim=0)
    perm  = torch.randperm(len(out_f))
    return out_f[perm], out_l[perm]


# ================================================================
#  GET DATA LOADER — LOOCV AWARE
#  FIX 4: Returns train_feats and train_labels in addition to the
#          DataLoader and test set, so train_journal.py can correctly
#          pass training data to evaluate_downstream().
# ================================================================
def get_loocv_loader(all_features: list,
                     all_labels: list,
                     test_idx: int,
                     batch_size: int = BATCH_SIZE,
                     noise_level: float = 0.0,
                     do_oversample: bool = True) -> tuple:
    """
    Build DataLoader for one LOOCV fold.

    Args:
        all_features  : list of feature tensors from load_all_subjects()
        all_labels    : list of label tensors from load_all_subjects()
        test_idx      : held-out subject index
        batch_size    : training batch size
        noise_level   : Gaussian noise std added to training features
        do_oversample : oversample minority classes before training

    Returns:
        train_loader : DataLoader for generative model training (features only)
        train_feats  : Tensor [n_train, n_features]  ← NEW (FIX 4)
        train_labels : Tensor [n_train]               ← NEW (FIX 4)
        test_feats   : Tensor [n_test, n_features]
        test_labels  : Tensor [n_test] stage labels for downstream evaluation
    """
    train_f, train_l, test_f, test_l = loocv_split(
        all_features, all_labels, test_idx
    )

    if do_oversample:
        train_f, train_l = oversample_minority(train_f, train_l)

    if noise_level > 0.0:
        noise   = torch.randn_like(train_f) * noise_level
        train_f = torch.clamp(train_f + noise, -1.0, 1.0)

    loader = DataLoader(
        TensorDataset(train_f),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    # Return train_feats and train_labels so evaluate_downstream gets the
    # correct split (FIX 4)
    return loader, train_f, train_l, test_f, test_l


# ================================================================
#  AUGMENTED DATASET BUILDER — for downstream classifier evaluation
#  FIX 2: Dynamic minority class detection
# ================================================================
def build_augmented_dataset(real_feats: torch.Tensor,
                             real_labels: torch.Tensor,
                             generator,
                             target_stage: int = None,
                             n_synthetic: int = None) -> tuple:
    """
    Generate synthetic minority-class samples using a trained generator,
    concatenate with real data for downstream SVM/RF training.

    FIX 2: target_stage is now detected dynamically from real_labels
    instead of being hardcoded to 1 (N1). On some subjects N3 or REM
    can be rarer than N1.

    Args:
        real_feats    : [n, n_features] real EEG features
        real_labels   : [n] integer stage labels
        generator     : trained generator model (quantum or classical)
        target_stage  : stage to augment. If None, uses the rarest class.
        n_synthetic   : samples to generate (default: balance to majority count)

    Returns:
        aug_feats  : [n + n_synthetic, n_features]
        aug_labels : [n + n_synthetic]
    """
    generator.eval()
    n_features = real_feats.shape[1]

    unique, counts = real_labels.unique(return_counts=True)
    max_count      = int(counts.max())

    # FIX 2: find minority class dynamically
    if target_stage is None:
        min_idx     = int(counts.argmin())
        target_stage = int(unique[min_idx])

    target_count = int((real_labels == target_stage).sum())

    if n_synthetic is None:
        n_synthetic = max(0, max_count - target_count)

    if n_synthetic == 0:
        generator.train()
        return real_feats, real_labels

    with torch.no_grad():
        z         = torch.randn(n_synthetic, n_features)
        synthetic = generator(z)
        synthetic = synthetic[:, :n_features]
        synthetic = torch.clamp(synthetic, -1.0, 1.0)

    generator.train()

    syn_labels = torch.full((n_synthetic,), target_stage, dtype=torch.long)
    aug_feats  = torch.cat([real_feats,  synthetic],  dim=0)
    aug_labels = torch.cat([real_labels, syn_labels], dim=0)
    return aug_feats, aug_labels


# ================================================================
#  SUBJECT CHECKER — utility
# ================================================================
def check_subjects(paths: list = None, feature_set: str = "statistical"):
    if paths is None:
        paths = SUBJECT_PATHS
    print(f"\n  {'='*65}")
    print(f"  SUBJECT CHECK | feature_set={feature_set}")
    print(f"  {'─'*65}")
    valid = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  {os.path.basename(path):<30}  FILE NOT FOUND")
            continue
        try:
            raw, labs = _load_subject_raw(path, feature_set=feature_set)
            n = raw.shape[0]
            unique_stages = np.unique(labs).tolist()
            print(f"  {os.path.basename(path):<30}  {n} epochs  "
                  f"stages={unique_stages}")
            if n >= 50:
                valid.append(path)
        except Exception as e:
            print(f"  {os.path.basename(path):<30}  ERROR: {e}")
    print(f"  {'─'*65}")
    print(f"  Valid: {len(valid)}/{len(paths)} | LOOCV folds: {len(valid)}")
    print(f"  {'='*65}\n")
    return valid


if __name__ == "__main__":
    check_subjects()