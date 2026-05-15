# qgan/data_loader_journal.py
# Multi-subject data loader with LOOCV — ANPHY-Sleep dataset
#
# ═══════════════════════════════════════════════════════════════════════════════
#  FINAL FIX: Hypnogram text files are tab‑separated: stage<TAB>onset<TAB>duration
#  e.g. "W\t0\t30" or "N1\t60\t30". The parser now reads the first token
#  and maps it to AASM integer stage (0=W,1=N1,2=N2,3=N3,4=REM).
#  All existing caching, global normalization, and LOOCV logic is preserved.
# ═══════════════════════════════════════════════════════════════════════════════

import os
import re
import csv
import xml.etree.ElementTree as ET
import warnings
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyedflib

from qgan.config import EEG_CHANNEL, EPOCH_SECONDS, BATCH_SIZE

# ================================================================
#  SUBJECT FILE PATHS – will be filtered to existing files at runtime
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

FREQ_BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "sigma": (12.0, 15.0),
    "beta":  (15.0, 30.0),
}

_RAW_SUBJECT_CACHE: dict = {}

# Stage mapping (AASM standard)
_STRING_STAGE_MAP = {
    "W":    0, "WAKE": 0, "0": 0,
    "N1":   1, "1":    1, "S1": 1,
    "N2":   2, "2":    2, "S2": 2,
    "N3":   3, "3":    3, "S3": 3, "S4": 3, "4": 3,
    "REM":  4, "R":    4, "5":  4,
    "?":   -1, "M":   -1, "X": -1,
}
_PROFUSION_STAGE_MAP = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 3, "5": 4, "9": -1,
}

# -----------------------------------------------------------------
#  IMPROVED TEXT HYPNOGRAM PARSER (supports single‑column & three‑column)
# -----------------------------------------------------------------
def _load_text_hypnogram(txt_path: str) -> np.ndarray:
    """
    Load sleep stages from a plain‑text hypnogram file.
    Supports:
        - One stage per line: "W", "N1", "2", etc.
        - Three‑column tab‑separated: "W\t0\t30", "N1\t60\t30"
    Blank lines and comment lines (#) are skipped.
    """
    labels = []
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Split by whitespace (tabs or spaces)
            parts = line.split()
            if not parts:
                continue
            token = parts[0].upper()   # first token = stage label
            if token in _STRING_STAGE_MAP:
                labels.append(_STRING_STAGE_MAP[token])
            else:
                # Try as integer code (Profusion style)
                try:
                    code = str(int(token))
                    labels.append(_PROFUSION_STAGE_MAP.get(code, -1))
                except ValueError:
                    labels.append(-1)
    return np.array(labels, dtype=np.int64)

# -----------------------------------------------------------------
#  XML, CSV, annotation EDF loaders (unchanged, kept from original)
# -----------------------------------------------------------------
def _load_profusion_xml(xml_path: str) -> np.ndarray:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    stage_tags = ["Stage", "Epoch", "SleepStage", "stage", "epoch"]
    stages = []
    for tag in stage_tags:
        found = root.findall(f".//{tag}")
        if found:
            stages = found
            break
    if not stages:
        raise ValueError(f"Cannot find stage elements in {xml_path}")
    labels = []
    for s in stages:
        code = str(s.text).strip() if s.text else "9"
        labels.append(_PROFUSION_STAGE_MAP.get(code, -1))
    return np.array(labels, dtype=np.int64)

def _load_csv_stages(csv_path: str) -> np.ndarray:
    labels = []
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    header = rows[0]
    stage_col = None
    stage_kws = ["stage", "label", "class", "score", "sleep", "annotation"]
    for i, h in enumerate(header):
        if any(kw in h.lower() for kw in stage_kws):
            stage_col = i
            break
    data_start = 0
    if stage_col is None:
        for i, val in enumerate(rows[0]):
            try:
                int(val.strip())
                stage_col = i
                break
            except ValueError:
                continue
        if stage_col is None:
            stage_col = 0
    else:
        data_start = 1
    for row in rows[data_start:]:
        if not row or len(row) <= stage_col:
            continue
        token = str(row[stage_col]).strip().upper()
        if token in _STRING_STAGE_MAP:
            labels.append(_STRING_STAGE_MAP[token])
        else:
            try:
                code = str(int(token))
                labels.append(_PROFUSION_STAGE_MAP.get(code, -1))
            except ValueError:
                continue
    return np.array(labels, dtype=np.int64)

def _load_annotation_edf(ann_edf_path: str, n_epochs: int) -> np.ndarray:
    labels = np.zeros(n_epochs, dtype=np.int64)
    stage_re = re.compile(r"sleep\s+stage\s+(\S+)", re.IGNORECASE)
    bare_re = re.compile(r"^(W|WAKE|N1|N2|N3|N4|R|REM|\?)$", re.IGNORECASE)
    token_map = {"W":0,"WAKE":0,"1":1,"N1":1,"2":2,"N2":2,"3":3,"N3":3,"4":3,"N4":3,"R":4,"REM":4,"5":4,"?":-1}
    with pyedflib.EdfReader(ann_edf_path) as f:
        try:
            ann = f.readAnnotations()
            ann_onsets, ann_descs = ann[0], ann[2]
        except Exception as e:
            raise ValueError(f"Cannot read annotations from {ann_edf_path}: {e}")
    for onset, desc in zip(ann_onsets, ann_descs):
        desc_s = str(desc).strip()
        label = None
        m = stage_re.search(desc_s)
        if m:
            token = m.group(1).upper()
            label = token_map.get(token)
        elif bare_re.match(desc_s):
            token = desc_s.upper()
            label = token_map.get(token)
        if label is None:
            continue
        idx = int(round(float(onset) / EPOCH_SECONDS))
        if 0 <= idx < n_epochs:
            labels[idx] = label if label >= 0 else -1
    return labels

# -----------------------------------------------------------------
#  IMPROVED SCORING FILE DETECTION (case‑insensitive, more suffixes)
# -----------------------------------------------------------------
def find_scoring_file(edf_path: str) -> tuple:
    """
    Search for a companion scoring file for the given EDF.
    Returns (scoring_file_path, format_string) or (None, None).
    """
    stem = os.path.splitext(os.path.basename(edf_path))[0]
    base = os.path.dirname(os.path.abspath(edf_path))
    parent = os.path.dirname(base)
    search_dirs = [
        base, parent,
        os.path.join(base, "scoring"), os.path.join(base, "labels"),
        os.path.join(base, "hypnograms"), os.path.join(base, "annotations"),
        os.path.join(parent, "scoring"), os.path.join(parent, "labels"),
        os.path.join(parent, "hypnograms"), os.path.join(parent, "annotations"),
    ]
    # (suffix, format_tag, case_sensitive?) – we will do case‑insensitive search
    candidates = [
        ("_PROF.xml", "profusion_xml"), ("-profusion.xml", "profusion_xml"),
        ("_profusion.xml", "profusion_xml"), (".xml", "profusion_xml"),
        ("_hypnogram.txt", "text"), ("_Hypnogram.txt", "text"),
        ("_stages.txt", "text"), ("_labels.txt", "text"),
        (".txt", "text"),
        ("_stages.csv", "csv"), ("_labels.csv", "csv"),
        ("_hypnogram.csv", "csv"), (".csv", "csv"),
        ("-annotations.edf", "annotation_edf"), ("_annotations.edf", "annotation_edf"),
        ("-Hypnogram.edf", "annotation_edf"), ("_hypnogram.edf", "annotation_edf"),
        ("Hypnogram.edf", "annotation_edf"),
        ("_events.tsv", "csv"), ("_staging.tsv", "csv"),
    ]
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for suffix, fmt in candidates:
            # Exact match
            path = os.path.join(search_dir, stem + suffix)
            if os.path.exists(path):
                return path, fmt
            # Case‑insensitive match (for Windows/Linux)
            try:
                for f in os.listdir(search_dir):
                    if f.lower() == (stem + suffix).lower():
                        return os.path.join(search_dir, f), fmt
            except OSError:
                pass
    return None, None

def load_scoring_file(edf_path: str, n_epochs: int) -> tuple:
    """
    Load sleep stage labels for one subject.
    Returns (label_array, source_description).
    If no scoring file found, returns all‑zeros (Wake) with a warning.
    """
    scoring_path, fmt = find_scoring_file(edf_path)
    name = os.path.basename(edf_path)
    if scoring_path is None:
        warnings.warn(
            f"\n  ╔══════════════════════════════════════════════════════════╗\n"
            f"  ║  NO SCORING FILE FOUND for {name:<30}    ║\n"
            f"  ║  Falling back to UNSUPERVISED mode (all epochs=Wake).  ║\n"
            f"  ╚══════════════════════════════════════════════════════════╝",
            stacklevel=3,
        )
        return np.zeros(n_epochs, dtype=np.int64), "UNSUPERVISED (no scoring file)"
    try:
        if fmt == "profusion_xml":
            labels = _load_profusion_xml(scoring_path)
        elif fmt == "text":
            labels = _load_text_hypnogram(scoring_path)
        elif fmt == "csv":
            labels = _load_csv_stages(scoring_path)
        elif fmt == "annotation_edf":
            labels = _load_annotation_edf(scoring_path, n_epochs)
        else:
            raise ValueError(f"Unknown format: {fmt}")
        # Align length to n_epochs
        if len(labels) < n_epochs:
            pad = np.zeros(n_epochs - len(labels), dtype=np.int64)
            labels = np.concatenate([labels, pad])
        elif len(labels) > n_epochs:
            labels = labels[:n_epochs]
        source = f"{os.path.basename(scoring_path)} ({fmt})"
        return labels, source
    except Exception as e:
        warnings.warn(f"{name}: Failed to load {scoring_path}: {e}. Using all‑Wake.", stacklevel=3)
        return np.zeros(n_epochs, dtype=np.int64), f"ERROR: {e}"

# -----------------------------------------------------------------
#  FEATURE EXTRACTION, NORMALIZATION, LOADER (unchanged, kept intact)
# -----------------------------------------------------------------
def _band_power(epoch: np.ndarray, fs: float, low: float, high: float) -> float:
    try:
        from scipy.signal import welch
    except ImportError:
        raise ImportError("scipy required: pip install scipy")
    n = len(epoch)
    nperseg = min(256, n // 4)
    if nperseg < 16:
        return 0.0
    freqs, psd = welch(epoch, fs=fs, window="hann", nperseg=nperseg)
    mask = (freqs >= low) & (freqs < high)
    return float(np.mean(psd[mask])) if np.any(mask) else 0.0

def _extract_features(epoch: np.ndarray, fs: float, feature_set: str = "statistical") -> np.ndarray:
    feats = []
    if feature_set in ("statistical", "combined"):
        feats += [epoch.mean(), epoch.std(), epoch.min(), epoch.max()]
    if feature_set in ("spectral", "combined"):
        for _, (lo, hi) in FREQ_BANDS.items():
            feats.append(_band_power(epoch, fs, lo, hi))
    if feature_set not in ("statistical", "spectral", "combined"):
        raise ValueError(f"Unknown feature_set='{feature_set}'.")
    return np.array(feats, dtype=np.float32)

def _n_features_for_set(feature_set: str) -> int:
    return {"statistical": 4, "spectral": 5, "combined": 9}[feature_set]

def _load_subject_raw(path: str, feature_set: str = "statistical", channel_idx: int = None) -> tuple:
    cache_key = (path, feature_set, channel_idx)
    if cache_key in _RAW_SUBJECT_CACHE:
        return _RAW_SUBJECT_CACHE[cache_key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"EDF not found: {path}")
    chan = channel_idx if channel_idx is not None else EEG_CHANNEL
    with pyedflib.EdfReader(path) as f:
        signal = f.readSignal(chan)
        sample_rate = f.getSampleFrequency(chan)
    samples_per_epoch = int(EPOCH_SECONDS * sample_rate)
    n_epochs = len(signal) // samples_per_epoch
    if n_epochs < 50:
        raise ValueError(f"{path}: only {n_epochs} epochs (need >= 50).")
    signal = signal[:n_epochs * samples_per_epoch]
    epochs_raw = signal.reshape(n_epochs, samples_per_epoch)
    stage_labels, _ = load_scoring_file(path, n_epochs)
    valid = stage_labels != -1
    epochs_raw = epochs_raw[valid]
    stage_labels = stage_labels[valid]
    n_valid = len(epochs_raw)
    if n_valid < 50:
        raise ValueError(f"{path}: only {n_valid} valid epochs after exclusions.")
    expected_n = _n_features_for_set(feature_set)
    feat_matrix = np.zeros((n_valid, expected_n), dtype=np.float32)
    for i, epoch in enumerate(epochs_raw):
        raw = _extract_features(epoch, sample_rate, feature_set)
        feat_matrix[i] = raw[:expected_n]
    _RAW_SUBJECT_CACHE[cache_key] = (feat_matrix, stage_labels)
    return feat_matrix, stage_labels

def _normalize_globally(raw_matrices: list) -> tuple:
    all_data = np.vstack(raw_matrices)
    col_min = all_data.min(axis=0)
    col_max = all_data.max(axis=0)
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
    return normalized, {"global_min": col_min.tolist(), "global_max": col_max.tolist()}

def load_all_subjects(n_features: int = 4, feature_set: str = "statistical", paths: list = None) -> tuple:
    if paths is None:
        paths = SUBJECT_PATHS
    available = [p for p in paths if os.path.exists(p)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        warnings.warn(f"Skipping {len(missing)} missing EDF file(s): {missing}")
    if not available:
        raise FileNotFoundError("No EDF files found. Check SUBJECT_PATHS.")
    print(f"\n  {'='*70}")
    print(f"  LOADING SUBJECTS | feature_set={feature_set} | n_features={n_features}")
    print(f"  Subjects: {len(available)} available / {len(paths)} listed")
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
        raise ValueError(f"Need >= 2 valid subjects for LOOCV. Got {len(raw_matrices)}.")
    normalized_matrices, scaler = _normalize_globally(raw_matrices)
    all_features_list, all_labels_list = [], []
    for name, norm, labs in zip(names, normalized_matrices, label_arrays):
        feat_t = torch.from_numpy(norm.astype(np.float32))
        label_t = torch.from_numpy(labs)
        all_features_list.append(feat_t)
        all_labels_list.append(label_t)
        stage_counts = {s: int((label_t == v).sum()) for s, v in [("W",0),("N1",1),("N2",2),("N3",3),("REM",4)] if int((label_t == v).sum()) > 0}
        print(f"    {name:<25} {feat_t.shape[0]:>4} epochs | {stage_counts}")
    total = sum(t.shape[0] for t in all_features_list)
    cat_labels = torch.cat(all_labels_list)
    print(f"  {'─'*70}\n  Total: {total} epochs across {len(all_features_list)} subjects")
    print(f"\n  Stage distribution (all subjects combined):")
    all_wake = True
    for sname, sint in [("W",0),("N1",1),("N2",2),("N3",3),("REM",4)]:
        cnt = int((cat_labels == sint).sum())
        if cnt > 0:
            pct = 100.0 * cnt / len(cat_labels)
            bar = "=" * int(pct / 2)
            print(f"    {sname:<5} {cnt:>5} ({pct:5.1f}%)  {bar}")
            if sint != 0:
                all_wake = False
    if all_wake:
        print(f"\n  ⚠️  WARNING: All epochs are Wake — scoring files not found!")
    print(f"  {'='*70}\n")
    return all_features_list, all_labels_list, scaler

# -----------------------------------------------------------------
#  LOOCV SPLIT, OVERSAMPLING, DATALOADER (unchanged)
# -----------------------------------------------------------------
def loocv_split(all_features: list, all_labels: list, test_idx: int) -> tuple:
    train_f = torch.cat([f for i, f in enumerate(all_features) if i != test_idx])
    train_l = torch.cat([l for i, l in enumerate(all_labels) if i != test_idx])
    return train_f, train_l, all_features[test_idx], all_labels[test_idx]

def oversample_minority(features: torch.Tensor, labels: torch.Tensor) -> tuple:
    unique, counts = labels.unique(return_counts=True)
    max_count = int(counts.max())
    feat_parts = [features]
    label_parts = [labels]
    for stage, count in zip(unique, counts):
        shortfall = max_count - int(count)
        if shortfall <= 0:
            continue
        mask = (labels == stage)
        sf = features[mask]
        idx = torch.randint(0, int(count), (shortfall,))
        feat_parts.append(sf[idx])
        label_parts.append(labels[mask][idx])
    out_f = torch.cat(feat_parts, dim=0)
    out_l = torch.cat(label_parts, dim=0)
    perm = torch.randperm(len(out_f))
    return out_f[perm], out_l[perm]

def get_loocv_loader(all_features: list, all_labels: list, test_idx: int,
                     batch_size: int = BATCH_SIZE, noise_level: float = 0.0,
                     do_oversample: bool = True) -> tuple:
    train_f, train_l, test_f, test_l = loocv_split(all_features, all_labels, test_idx)
    if do_oversample:
        train_f, train_l = oversample_minority(train_f, train_l)
    if noise_level > 0.0:
        noise = torch.randn_like(train_f) * noise_level
        train_f = torch.clamp(train_f + noise, -1.0, 1.0)
    loader = DataLoader(TensorDataset(train_f), batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, train_f, train_l, test_f, test_l

def build_augmented_dataset(real_feats: torch.Tensor, real_labels: torch.Tensor,
                            generator, target_stage: int = None, n_synthetic: int = None) -> tuple:
    generator.eval()
    n_features = real_feats.shape[1]
    unique, counts = real_labels.unique(return_counts=True)
    max_count = int(counts.max())
    if target_stage is None:
        target_stage = int(unique[int(counts.argmin())])
    target_count = int((real_labels == target_stage).sum())
    if n_synthetic is None:
        n_synthetic = max(0, max_count - target_count)
    if n_synthetic == 0:
        generator.train()
        return real_feats, real_labels
    with torch.no_grad():
        z = torch.randn(n_synthetic, n_features)
        synthetic = generator(z)[:, :n_features]
        synthetic = torch.clamp(synthetic, -1.0, 1.0)
    generator.train()
    syn_labels = torch.full((n_synthetic,), target_stage, dtype=torch.long)
    return (torch.cat([real_feats, synthetic], dim=0),
            torch.cat([real_labels, syn_labels], dim=0))

def find_all_scoring_files(paths: list = None):
    if paths is None:
        paths = SUBJECT_PATHS
    print(f"\n  {'='*70}\n  SCORING FILE SEARCH\n  {'─'*70}")
    found = 0
    for path in paths:
        name = os.path.basename(path)
        if not os.path.exists(path):
            print(f"  ⚠️ {name} – EDF missing")
            continue
        scoring_path, fmt = find_scoring_file(path)
        if scoring_path:
            print(f"  ✅ {name:<25} → {os.path.basename(scoring_path)} ({fmt})")
            found += 1
        else:
            print(f"  ❌ {name:<25} → NOT FOUND")
    print(f"  {'─'*70}\n  Found: {found}/{len(paths)} scoring files\n  {'='*70}\n")
    return found

if __name__ == "__main__":
    find_all_scoring_files()