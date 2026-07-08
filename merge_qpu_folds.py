# merge_qpu_folds.py
# NEW: stitches per-fold JSON files (produced when ONLY_FOLD was set) back
# into the single results_<condition>_<feature_set>.json shape that
# compute_significance_tests() and _print_summary_table() expect.
#
# Usage:
#   python merge_qpu_folds.py qpu_sim statistical
#   python merge_qpu_folds.py qpu_sim spectral

import json
import sys
from qgan.train_journal import (
    aggregate_folds, _atomic_write, out_file_path, FEATURE_SET_N,
)


def merge(condition: str, feature_set: str, n_folds: int = 10) -> None:
    merged_folds = []
    n_features = FEATURE_SET_N.get(feature_set)

    for fold_idx in range(n_folds):
        fname = out_file_path(condition, feature_set, fold_idx=fold_idx)
        try:
            with open(fname) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"  missing: {fname}")
            continue

        for feat_key, cdata in data.items():
            merged_folds.extend(cdata.get("folds", []))

    if not merged_folds:
        print(f"  No folds found for {condition}/{feature_set} — nothing to merge.")
        return

    # de-dupe by fold_idx in case a fold file was re-run
    by_idx = {f["fold_idx"]: f for f in merged_folds}
    merged_folds = [by_idx[i] for i in sorted(by_idx)]

    feat_key = f"{feature_set}_{n_features}feat"
    out = {
        feat_key: {
            "n_features":     n_features,
            "feature_set":    feature_set,
            "condition":      condition,
            "generator_type": "quantum",
            "folds":          merged_folds,
            "aggregated":     aggregate_folds(merged_folds),
        }
    }

    final_name = out_file_path(condition, feature_set)
    _atomic_write(final_name, out)
    print(f"  Merged {len(merged_folds)} folds -> {final_name}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_qpu_folds.py <condition> <feature_set> [n_folds]")
        sys.exit(1)
    condition   = sys.argv[1]
    feature_set = sys.argv[2]
    n_folds     = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    merge(condition, feature_set, n_folds)