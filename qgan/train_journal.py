# qgan/train_journal.py
# Journal Extension — Four-Backend LOOCV Training
#
# ── HOW TO RUN ───────────────────────────────────────────────────────────────
#
# LOCAL SMOKE TEST (verify code works, ~10 min):
#   python -m qgan.train_journal --mode local
#
# ARC — CPU CONDITIONS (Conditions 1+2):
#   CPU_EPOCHS=50 python -m qgan.train_journal --mode full --conditions cpu
#
# ARC — QPU CONDITIONS (Conditions 3+4):
#   QPU_EPOCHS=100 python -m qgan.train_journal --mode full --conditions qpu
#
# ARC — ALL CONDITIONS:
#   CPU_EPOCHS=50 QPU_EPOCHS=100 python -m qgan.train_journal --mode full --conditions all
#
# RESUME (crashed job — same command, auto-detects last completed fold):
#   python -m qgan.train_journal --mode full --conditions qpu
#   No separate resume flag needed. The checkpoint system skips all
#   completed folds and continues from the first pending one.
#
# ── CHANGES FROM PREVIOUS VERSION ────────────────────────────────────────────
#
#  RESUME BUG — FIXED (was restarting from fold 0 every time)
#    OLD: completed_folds = len(valid_folds)
#         if fold_idx < completed_folds: skip
#    PROBLEM: Positional count. Any rejected fold collapses the range.
#             If fold 0 fails validation: completed_folds=0, all reruns.
#    FIX: valid_indices = {fold["fold_idx"] for fold in valid_folds}
#         if fold_idx in valid_indices: skip
#         Each fold is matched by its actual index, not list position.
#
#  ATOMIC CHECKPOINT WRITES — ADDED
#    OLD: directly overwrite results file after each fold
#    PROBLEM: On ARC, if the job is killed mid-write, the JSON is
#             truncated/corrupt and the entire condition must restart.
#    FIX: Write to <file>.tmp, fsync, then os.replace (atomic rename).
#         A killed job always leaves a complete prior checkpoint.
#
#  ETA ESTIMATION — ADDED
#    Tracks rolling average of fold wall-clock times.
#    Prints "ETA: Xh Ym remaining" after each fold.
#    Helps decide whether to extend the SLURM time limit.
#
#  WALL-CLOCK BUDGET GUARD — ADDED
#    SLURM_JOB_END_TIME env var (set in job script) lets the training
#    loop detect an approaching wall-clock deadline and stop cleanly
#    rather than being killed mid-fold (which would corrupt nothing
#    thanks to atomic writes, but wastes partial fold computation).
#
#  COLLAPSED FOLD DETECTION — ADDED
#    Folds with F1 < 0.15 are flagged as collapsed in the checkpoint
#    and summary table. Not auto-retrained — reported and discussed
#    in the paper (generator failure modes are scientifically relevant).
#
#  EPOCH COUNT VALIDATION — ADDED
#    Folds with n_epochs_trained < MIN_EPOCHS_TO_ACCEPT are rejected
#    and retrained. Prevents smoke-test checkpoints from poisoning
#    the full-training resume.
#
# ── OUTPUT FILES ─────────────────────────────────────────────────────────────
#   results_journal_simulator.json
#   results_journal_simulator_datanoise.json
#   results_journal_qpu_noiseless.json
#   results_journal_qpu_noise_zne.json
#   results_journal_SUMMARY.json   ← combined summary for paper tables

import argparse
import copy
import json
import os
import sys
import tempfile
import time
import warnings
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

from qgan.config import LEARNING_RATE, GRAD_CLIP, BATCH_SIZE
from qgan.data_loader_journal import (
    load_all_subjects, get_loocv_loader, SUBJECT_PATHS
)
from qgan.models_journal import (
    build_models,
    ALL_CONDITIONS,
    CONDITION_SIMULATOR,
    CONDITION_DATA_NOISE,
    CONDITION_QPU_SIM,
    CONDITION_QPU_ZNE,
    CONDITION_LABELS,
    CONDITION_GRAD_METHOD,
    QPU_SHOTS,
)


# ================================================================
#  EXPERIMENT CONFIG
# ================================================================
N_QUBITS    = 6
LAMBDA_GP   = 10
NOISE_LEVEL = 0.1

LR_G = LEARNING_RATE
LR_D = LEARNING_RATE * 5.0

# Folds trained for fewer than this many epochs are considered incomplete.
# Rejects smoke-test (--mode local) folds from full-training checkpoints.
MIN_EPOCHS_TO_ACCEPT = 10

# F1 below this threshold = generator collapsed (failed to learn distribution).
# Flagged in summary and checkpoint. NOT auto-retrained.
COLLAPSE_F1_THRESHOLD = 0.15

# Seconds before SLURM wall-clock limit to stop cleanly.
# Set SLURM_JOB_END_TIME in your job script to enable this guard.
WALL_CLOCK_BUFFER_S = 600   # 10 min safety margin

CONDITION_GROUPS = {
    "cpu": [CONDITION_SIMULATOR, CONDITION_DATA_NOISE],
    "qpu": [CONDITION_QPU_SIM,   CONDITION_QPU_ZNE],
    "all": ALL_CONDITIONS,
}

OUT_FILES = {
    CONDITION_SIMULATOR:  "results_journal_simulator.json",
    CONDITION_DATA_NOISE: "results_journal_simulator_datanoise.json",
    CONDITION_QPU_SIM:    "results_journal_qpu_noiseless.json",
    CONDITION_QPU_ZNE:    "results_journal_qpu_noise_zne.json",
}
SUMMARY_FILE = "results_journal_SUMMARY.json"


# ================================================================
#  UTILITIES
# ================================================================

def _seconds_until_wall_limit() -> float:
    """
    Returns seconds remaining before SLURM wall-clock limit.
    Requires SLURM_JOB_END_TIME (Unix timestamp) to be set in the job script.
    Returns float('inf') if not set.
    """
    end_time = os.getenv("SLURM_JOB_END_TIME")
    if end_time is None:
        return float("inf")
    try:
        return float(end_time) - time.time()
    except ValueError:
        return float("inf")


def _fmt_seconds(s: float) -> str:
    """Format a duration in seconds as 'Xh Ym' or 'Ym Zs'."""
    s = max(0.0, s)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def _atomic_write(path: str, data: dict) -> None:
    """
    Write JSON to path atomically using write-to-tmp + fsync + rename.

    On ARC, SLURM kills jobs with SIGKILL — not SIGTERM — so there is
    no graceful shutdown. A direct overwrite can produce a half-written
    JSON if the kill lands during the write. Atomic rename guarantees
    the checkpoint file is always either the previous complete state
    or the new complete state, never a corrupt intermediate.
    """
    dir_name  = os.path.dirname(os.path.abspath(path)) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)   # atomic on POSIX (Lustre included)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ================================================================
#  WGAN-GP GRADIENT PENALTY
# ================================================================
def gradient_penalty(disc, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    bs    = real.size(0)
    alpha = torch.rand(bs, 1, dtype=torch.float32).expand_as(real)
    interp = (alpha * real.float() +
              (1 - alpha) * fake.float()).requires_grad_(True)
    d_out = disc(interp)
    grads = torch.autograd.grad(
        outputs=d_out, inputs=interp,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grads.view(bs, -1).norm(2, dim=1) - 1) ** 2).mean()


# ================================================================
#  METRICS
# ================================================================
def compute_mae(gen, data_tensor: torch.Tensor, n_features: int) -> dict:
    """Mean and std MAE between real and generated feature distributions."""
    gen.eval()
    n = min(len(data_tensor), 100)
    with torch.no_grad():
        real     = data_tensor[:n, :n_features].float()
        z        = torch.randn(n, n_features)
        fake_all = gen(z)
        fake     = fake_all[:, :n_features]
    gen.train()
    return {
        "mean_MAE": round(torch.abs(real.mean(0) - fake.mean(0)).mean().item(), 4),
        "std_MAE":  round(torch.abs(real.std(0)  - fake.std(0) ).mean().item(), 4),
    }


def compute_clf(gen, disc, data_tensor: torch.Tensor, n_features: int) -> dict:
    """
    Classification metrics treating discriminator as a real/fake classifier.

    Real samples → label 1. Generated samples → label 0.
    Threshold = 0.0 (WGAN-GP score boundary).

    Returns: Accuracy, Precision, Sensitivity (Recall), Specificity, F1.
    """
    gen.eval(); disc.eval()
    n = min(len(data_tensor), 100)
    with torch.no_grad():
        real = data_tensor[:n, :n_features].float()
        z    = torch.randn(n, n_features)
        fake = gen(z)

        if real.shape[-1] < N_QUBITS:
            pad     = torch.zeros(n, N_QUBITS - real.shape[-1])
            real_in = torch.cat([real, pad], dim=-1)
        else:
            real_in = real

        rs = disc(real_in).squeeze().detach().numpy()
        fs = disc(fake).squeeze().detach().numpy()

    scores = np.concatenate([rs, fs])
    labels = np.array([1] * n + [0] * n)
    preds  = (scores > 0.0).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    gen.train(); disc.train()
    return {
        "Accuracy":    round(accuracy_score(labels, preds), 4),
        "Precision":   round(precision_score(labels, preds, zero_division=0), 4),
        "Sensitivity": round(recall_score(labels, preds, zero_division=0), 4),
        "Specificity": round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
        "F1":          round(f1_score(labels, preds, zero_division=0), 4),
    }


# ================================================================
#  AGGREGATE LOOCV RESULTS
# ================================================================
def aggregate_folds(fold_results: list) -> dict:
    keys = ["mean_MAE", "std_MAE",
            "Accuracy", "Precision", "Sensitivity", "Specificity", "F1",
            "avg_time_per_epoch"]
    agg = {}
    for k in keys:
        vals = []
        for r in fold_results:
            if k in r.get("mae", {}):
                vals.append(r["mae"][k])
            elif k in r.get("clf", {}):
                vals.append(r["clf"][k])
            elif k in r.get("history", {}):
                vals.append(r["history"][k])
        if vals:
            agg[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)),  4),
                "all":  [round(v, 4) for v in vals],
            }
    return agg


# ================================================================
#  CHECKPOINT VALIDATION
# ================================================================
def get_valid_fold_indices(saved_folds: list) -> set:
    """
    Return the set of fold_idx values that passed epoch-count validation.

    Returns a SET so callers can do: if fold_idx in valid_indices: skip
    This is correct regardless of which or how many folds failed validation.
    A positional count (the old approach) breaks whenever any fold is
    rejected — causing wrong folds to be skipped or all folds to rerun.
    """
    valid_indices = set()
    for fold in saved_folds:
        idx       = fold.get("fold_idx")
        n_trained = fold.get("history", {}).get("n_epochs_trained", None)

        if idx is None:
            continue   # malformed fold entry

        if n_trained is None:
            # Legacy checkpoint: use timing heuristic
            avg_time = fold.get("history", {}).get("avg_time_per_epoch", 0)
            if avg_time >= 80:
                valid_indices.add(idx)
            else:
                print(f"  WARNING: fold {idx} — legacy checkpoint, "
                      f"avg_time={avg_time:.1f}s < 80s/epoch — will retrain.")
        elif n_trained >= MIN_EPOCHS_TO_ACCEPT:
            valid_indices.add(idx)
        else:
            print(f"  WARNING: fold {idx} — only {n_trained} epochs recorded "
                  f"(need >= {MIN_EPOCHS_TO_ACCEPT}) — will retrain.")
    return valid_indices


def check_collapsed_folds(saved_folds: list) -> list:
    """
    Return list of fold_idx values where F1 < COLLAPSE_F1_THRESHOLD.

    Collapsed folds trained successfully but the generator failed to
    learn the data distribution (mode collapse or vanishing gradient).
    They are retained in the checkpoint and flagged in the paper —
    generator failure modes under quantum noise are scientifically
    interesting, not just artifacts to discard.
    """
    return [
        fold.get("fold_idx")
        for fold in saved_folds
        if fold.get("clf", {}).get("F1", 1.0) < COLLAPSE_F1_THRESHOLD
    ]


# ================================================================
#  TRAINING LOOP — one LOOCV fold
# ================================================================
def train_one_fold(gen, disc, train_loader, test_data,
                   n_features: int, n_epochs: int, label: str) -> tuple:
    """
    WGAN-GP training for one LOOCV fold.

    Returns: (history dict, best_gen, best_disc)

    Generator is updated twice per discriminator update (2:1 ratio)
    with gradient clipping to prevent exploding gradients — standard
    for WGAN-GP with a quantum generator whose output is in [−1, +1].

    Best model state is tracked by discriminator loss (lower = better
    real/fake separation). Best states are restored before returning.
    """
    opt_g = torch.optim.Adam(gen.parameters(),  lr=LR_G, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR_D, betas=(0.0, 0.9))

    history = {
        "gen_loss":           [],
        "disc_loss":          [],
        "mean_MAE":           [],
        "std_MAE":            [],
        "mae_epochs":         [],
        "times":              [],
    }
    best_disc_loss  = float("inf")
    best_gen_state  = copy.deepcopy(gen.state_dict())
    best_disc_state = copy.deepcopy(disc.state_dict())

    fold_start = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()
        g_losses, d_losses = [], []

        for (batch,) in train_loader:
            real = batch.float()
            bs   = real.shape[0]

            if real.shape[-1] < N_QUBITS:
                pad     = torch.zeros(bs, N_QUBITS - real.shape[-1])
                real_in = torch.cat([real, pad], dim=-1)
            else:
                real_in = real

            # Generator update ×2
            for _ in range(2):
                z      = torch.randn(bs, n_features)
                fake   = gen(z)
                g_loss = -disc(fake).mean()
                opt_g.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP)
                opt_g.step()
                g_losses.append(g_loss.item())

            # Discriminator update ×1
            z    = torch.randn(bs, n_features)
            fake = gen(z).detach()
            gp   = gradient_penalty(disc, real_in, fake)
            d_loss = (-disc(real_in).mean()
                      + disc(fake).mean()
                      + LAMBDA_GP * gp)
            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), GRAD_CLIP)
            opt_d.step()
            d_losses.append(d_loss.item())

        avg_g   = float(np.mean(g_losses))
        avg_d   = float(np.mean(d_losses))
        elapsed = time.time() - epoch_start

        history["gen_loss"].append(avg_g)
        history["disc_loss"].append(avg_d)
        history["times"].append(elapsed)

        if avg_d < best_disc_loss:
            best_disc_loss  = avg_d
            best_gen_state  = copy.deepcopy(gen.state_dict())
            best_disc_state = copy.deepcopy(disc.state_dict())

        mae = compute_mae(gen, test_data, n_features)
        history["mean_MAE"].append(mae["mean_MAE"])
        history["std_MAE"].append(mae["std_MAE"])
        history["mae_epochs"].append(epoch + 1)

        # Log every 10 epochs, first epoch, and last epoch
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == n_epochs:
            elapsed_total = time.time() - fold_start
            epochs_done   = epoch + 1
            avg_per_epoch = elapsed_total / epochs_done
            remaining_ep  = n_epochs - epochs_done
            eta_fold      = _fmt_seconds(avg_per_epoch * remaining_ep)
            print(f"      Epoch [{epochs_done:3d}/{n_epochs}] "
                  f"G:{avg_g:+.4f}  D:{avg_d:+.4f}  "
                  f"StdMAE:{mae['std_MAE']:.4f}  "
                  f"Time:{elapsed:.1f}s  "
                  f"ETA(fold):{eta_fold}  [{label}]")

    gen.load_state_dict(best_gen_state)
    disc.load_state_dict(best_disc_state)
    history["avg_time_per_epoch"] = round(float(np.mean(history["times"])), 2)
    history["total_fold_time_s"]  = round(time.time() - fold_start, 1)
    history["n_epochs_trained"]   = n_epochs
    return history, gen, disc


# ================================================================
#  RUN ONE CONDITION
# ================================================================
def run_condition(condition: str,
                  feature_sweep: list,
                  n_epochs: int,
                  subject_paths: list) -> dict:

    label      = CONDITION_LABELS[condition]
    out_file   = OUT_FILES[condition]
    grad_method = CONDITION_GRAD_METHOD[condition]
    n_subjects = len(subject_paths)
    n_folds    = n_subjects

    print(f"\n  {'='*70}")
    print(f"  CONDITION : {label}")
    print(f"  Grad      : {grad_method}"
          + (f" (SPSA, h={0.05}, {QPU_SHOTS} shots)" if grad_method == "spsa" else " (exact)"))
    print(f"  Subjects  : {n_subjects}  |  LOOCV folds: {n_folds}")
    print(f"  Features  : {feature_sweep}  |  Epochs: {n_epochs}")
    print(f"  Output    : {out_file}")
    print(f"  {'='*70}")

    # Load checkpoint
    condition_results: dict = {}
    if os.path.exists(out_file):
        try:
            with open(out_file) as f:
                condition_results = json.load(f)
            print(f"  Loaded checkpoint: {out_file}")
        except json.JSONDecodeError:
            print(f"  WARNING: {out_file} is corrupt — starting fresh.")
            condition_results = {}

    noise_level = NOISE_LEVEL if condition == CONDITION_DATA_NOISE else 0.0

    # Rolling ETA tracker across all folds in this condition
    fold_times: list = []

    for n_features in feature_sweep:
        feat_key = f"{n_features}_features"

        if feat_key not in condition_results:
            condition_results[feat_key] = {
                "n_features": n_features,
                "condition":  condition,
                "folds":      [],
            }

        # ── Checkpoint validation ─────────────────────────────────────
        raw_folds = condition_results[feat_key].get("folds", [])

        # Build O(1) lookup by fold_idx
        saved_by_idx = {f["fold_idx"]: f for f in raw_folds if "fold_idx" in f}

        # Valid = passed epoch-count threshold
        valid_indices = get_valid_fold_indices(raw_folds)

        # Collapsed = trained but F1 too low
        collapsed_indices = check_collapsed_folds(
            [saved_by_idx[i] for i in valid_indices if i in saved_by_idx]
        )
        if collapsed_indices:
            print(f"\n  NOTICE: Collapsed folds for {feat_key}: "
                  f"{collapsed_indices} (F1 < {COLLAPSE_F1_THRESHOLD}). "
                  f"Kept and flagged — not auto-retrained.")

        # Rebuild fold list: only validated folds, sorted by index
        condition_results[feat_key]["folds"] = [
            saved_by_idx[i]
            for i in sorted(valid_indices)
            if i in saved_by_idx
        ]

        # ── Load data for this feature count ─────────────────────────
        print(f"\n  Loading {n_subjects} subjects | {n_features} features")
        all_data = load_all_subjects(n_features=n_features, paths=subject_paths)

        print(f"  {'─'*60}")
        for path, data in zip(subject_paths, all_data):
            print(f"    {os.path.basename(path):<28}  {len(data):>4} epochs")
        total_ep = sum(len(d) for d in all_data)
        print(f"  {'─'*60}")
        print(f"  Total: {total_ep} epochs across {n_subjects} subjects")

        # ── Resume status ─────────────────────────────────────────────
        pending = []
        print(f"\n  Resume status — {feat_key}:")
        for fold_idx in range(n_folds):
            subj = os.path.basename(subject_paths[fold_idx])
            if fold_idx in valid_indices:
                cflag = " ⚠ COLLAPSED" if fold_idx in collapsed_indices else ""
                print(f"    Fold {fold_idx:2d}  {subj:<28}  DONE{cflag}")
            else:
                pending.append(fold_idx)
                print(f"    Fold {fold_idx:2d}  {subj:<28}  PENDING")
        if pending:
            print(f"\n  Pending: {pending}  ({len(pending)} fold(s) remaining)")
        else:
            print(f"\n  All folds complete for {feat_key}. Skipping.")
            continue

        # ── Main fold loop ────────────────────────────────────────────
        for fold_idx in range(n_folds):

            # Core resume check — by index membership, not position
            if fold_idx in valid_indices:
                continue

            # Wall-clock budget guard
            remaining_wall = _seconds_until_wall_limit()
            if remaining_wall < WALL_CLOCK_BUFFER_S:
                print(f"\n  WALL-CLOCK GUARD: {_fmt_seconds(remaining_wall)} "
                      f"remaining (< {_fmt_seconds(WALL_CLOCK_BUFFER_S)} buffer). "
                      f"Stopping cleanly. Resubmit to continue.")
                return condition_results

            test_subject = os.path.basename(subject_paths[fold_idx])
            fold_wall_start = time.time()

            # ETA estimate based on rolling fold times
            if fold_times:
                avg_fold_t = float(np.mean(fold_times))
                remaining_folds = len([i for i in pending if i >= fold_idx])
                eta_str = _fmt_seconds(avg_fold_t * remaining_folds)
            else:
                eta_str = "estimating..."

            print(f"\n  ── {feat_key} | Fold {fold_idx+1}/{n_folds} "
                  f"(test: {test_subject}) | ETA remaining: {eta_str} ──")

            train_loader, test_data = get_loocv_loader(
                all_data, test_idx=fold_idx,
                batch_size=BATCH_SIZE,
                noise_level=noise_level,
            )
            n_train = sum(len(d) for i, d in enumerate(all_data) if i != fold_idx)
            print(f"    Train: {n_train} epochs  |  Test: {len(test_data)} epochs")

            gen, disc = build_models(condition, N_QUBITS, n_features)

            history, gen, disc = train_one_fold(
                gen, disc, train_loader, test_data,
                n_features, n_epochs,
                label=f"{label} | f={n_features} | fold={fold_idx+1}",
            )

            mae = compute_mae(gen, test_data, n_features)
            clf = compute_clf(gen, disc, test_data, n_features)

            is_collapsed = clf["F1"] < COLLAPSE_F1_THRESHOLD
            if is_collapsed:
                print(f"    WARNING: Fold {fold_idx} collapsed "
                      f"(F1={clf['F1']:.4f} < {COLLAPSE_F1_THRESHOLD}). "
                      f"Generator failed to learn distribution. Flagged.")

            fold_result = {
                "fold_idx":          fold_idx,
                "test_subject":      test_subject,
                "collapsed":         is_collapsed,
                "mae":               mae,
                "clf":               clf,
                "history": {
                    "n_epochs_trained":   n_epochs,
                    "avg_time_per_epoch": history["avg_time_per_epoch"],
                    "total_fold_time_s":  history["total_fold_time_s"],
                    "final_gen_loss":     round(history["gen_loss"][-1], 4),
                    "final_disc_loss":    round(history["disc_loss"][-1], 4),
                },
            }

            condition_results[feat_key]["folds"].append(fold_result)
            condition_results[feat_key]["folds"].sort(key=lambda f: f["fold_idx"])
            condition_results[feat_key]["aggregated"] = aggregate_folds(
                condition_results[feat_key]["folds"]
            )

            # Atomic write — safe against SLURM SIGKILL mid-write
            _atomic_write(out_file, condition_results)
            valid_indices.add(fold_idx)

            fold_wall_time = time.time() - fold_wall_start
            fold_times.append(fold_wall_time)

            collapsed_str = " ⚠ COLLAPSED" if is_collapsed else ""
            print(f"    Checkpoint saved (atomic): {out_file}  "
                  f"fold {fold_idx+1}/{n_folds}{collapsed_str}")
            print(f"    FOLD RESULT → "
                  f"Acc:{clf['Accuracy']:.4f}  "
                  f"Spec:{clf['Specificity']:.4f}  "
                  f"F1:{clf['F1']:.4f}  "
                  f"StdMAE:{mae['std_MAE']:.4f}  "
                  f"Time:{_fmt_seconds(fold_wall_time)}")

        # ── Aggregated result for this feature count ──────────────────
        agg = condition_results[feat_key].get("aggregated", {})
        collapsed_in_feat = check_collapsed_folds(
            condition_results[feat_key].get("folds", [])
        )
        if agg:
            print(f"\n  AGGREGATED — {n_features}f, {n_folds} folds"
                  + (f"  [{len(collapsed_in_feat)} collapsed fold(s) included]"
                     if collapsed_in_feat else ""))
            for metric in ["Accuracy", "Specificity", "F1", "std_MAE", "avg_time_per_epoch"]:
                if metric in agg:
                    m = agg[metric]
                    unit = "s" if metric == "avg_time_per_epoch" else ""
                    print(f"    {metric:<25}  {m['mean']:.4f} ± {m['std']:.4f}{unit}")

    print(f"\n  Condition complete: {out_file}")
    return condition_results


# ================================================================
#  MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Journal QGAN — 4-backend LOOCV training (ARC-optimised)"
    )
    parser.add_argument("--mode",       choices=["local", "full"], default="local")
    parser.add_argument("--conditions", choices=["cpu", "qpu", "all"], default="all")
    args = parser.parse_args()

    # Environment info (visible in ARC .log file)
    print(f"\n  {'='*70}")
    print(f"  JOURNAL QGAN — ARC TRAINING RUN")
    print(f"  Python  : {sys.version.split()[0]}")
    try:
        import pennylane as qml
        print(f"  PennyLane: {qml.__version__}")
    except ImportError:
        print(f"  PennyLane: NOT FOUND")
    try:
        import qiskit_aer
        print(f"  Qiskit Aer: {qiskit_aer.__version__}")
    except ImportError:
        print(f"  Qiskit Aer: NOT FOUND — QPU conditions will fall back to default.qubit")
    print(f"  OMP_NUM_THREADS : {os.getenv('OMP_NUM_THREADS', 'not set')}")
    slurm_job = os.getenv("SLURM_JOB_ID", "not set")
    slurm_node = os.getenv("SLURM_NODELIST", "not set")
    print(f"  SLURM job       : {slurm_job}")
    print(f"  SLURM nodes     : {slurm_node}")
    wall_remaining = _seconds_until_wall_limit()
    if wall_remaining < float("inf"):
        print(f"  Wall-clock remaining: {_fmt_seconds(wall_remaining)}")

    if args.mode == "local":
        base_epochs   = 3
        feature_sweep = [4]
        subject_paths = SUBJECT_PATHS[:3]
        print(f"\n  MODE: LOCAL SMOKE TEST")
        print(f"  Epochs=3 (auto-rejected by full-run checkpoint validation)")
        print(f"  Features=[4] | Subjects={len(subject_paths)}")
    else:
        cpu_epochs    = int(os.getenv("CPU_EPOCHS", "10"))
        qpu_epochs    = int(os.getenv("QPU_EPOCHS", "50"))
        feature_sweep = [2, 3, 4]
        subject_paths = SUBJECT_PATHS
        print(f"\n  MODE: FULL TRAINING RUN")
        print(f"  CPU_EPOCHS={cpu_epochs}  QPU_EPOCHS={qpu_epochs}")
        print(f"  Resume: rerun same command — completed folds skipped automatically.")

    conditions_to_run = CONDITION_GROUPS[args.conditions]

    print(f"\n  {'='*70}")
    print(f"  Subjects    : {len(subject_paths)} | LOOCV folds: {len(subject_paths)}")
    print(f"  Conditions  : {[CONDITION_LABELS[c] for c in conditions_to_run]}")
    print(f"  Features    : {feature_sweep}")
    print(f"  Architecture: Arch C — {N_QUBITS} qubits | ring CNOT | RX→CNOT→RY")
    print(f"  Loss        : WGAN-GP (lambda_gp={LAMBDA_GP})")
    print(f"  Min epochs to accept checkpoint : {MIN_EPOCHS_TO_ACCEPT}")
    print(f"  Collapse threshold (F1 flag)    : {COLLAPSE_F1_THRESHOLD}")
    print(f"  Checkpoint writes               : atomic (fsync + rename)")
    print(f"  {'='*70}")

    all_condition_results = {}
    run_start = time.time()

    for condition in conditions_to_run:
        if args.mode == "local":
            n_epochs = base_epochs
        elif condition in (CONDITION_SIMULATOR, CONDITION_DATA_NOISE):
            n_epochs = cpu_epochs
        else:
            n_epochs = qpu_epochs

        result = run_condition(
            condition=condition,
            feature_sweep=feature_sweep,
            n_epochs=n_epochs,
            subject_paths=subject_paths,
        )
        all_condition_results[condition] = result

    _atomic_write(SUMMARY_FILE, all_condition_results)
    total_run = time.time() - run_start
    print(f"\n  Combined summary saved: {SUMMARY_FILE}")
    print(f"  Total run time: {_fmt_seconds(total_run)}")
    _print_summary_table(all_condition_results, feature_sweep, conditions_to_run)


# ================================================================
#  SUMMARY TABLE
# ================================================================
def _print_summary_table(all_results: dict,
                          feature_sweep: list,
                          conditions: list) -> None:

    num_folds = 0
    for cond in conditions:
        if cond in all_results:
            keys = list(all_results[cond].keys())
            if keys:
                num_folds = len(all_results[cond][keys[0]].get("folds", []))
                break

    W = 100
    print(f"\n  {'='*W}")
    print(f"  FINAL RESULTS  (mean ± std across {num_folds} LOOCV folds)")
    print(f"  {'─'*W}")
    print(f"  {'Condition':<30} {'Feat':<8} {'Acc':<16} {'Spec':<16} "
          f"{'F1':<16} {'StdMAE':<16} {'Time/ep'}")
    print(f"  {'─'*W}")

    for condition in conditions:
        label = CONDITION_LABELS[condition]
        if condition not in all_results:
            continue
        for n_f in feature_sweep:
            feat_key = f"{n_f}_features"
            cond_data = all_results[condition].get(feat_key, {})
            agg = cond_data.get("aggregated", {})
            if not agg:
                print(f"  {label:<30} {n_f}f      (no data)")
                continue

            n_collapsed = len(check_collapsed_folds(cond_data.get("folds", [])))
            feat_str = f"{n_f}f" + (f" ⚠{n_collapsed}" if n_collapsed else "")

            def fmt(key):
                if key in agg:
                    return f"{agg[key]['mean']:.3f}±{agg[key]['std']:.3f}"
                return "---"

            print(f"  {label:<30} {feat_str:<8} "
                  f"{fmt('Accuracy'):<16} "
                  f"{fmt('Specificity'):<16} "
                  f"{fmt('F1'):<16} "
                  f"{fmt('std_MAE'):<16} "
                  f"{fmt('avg_time_per_epoch')}s")
        print(f"  {'─'*W}")

    print(f"  {'='*W}")
    print(f"\n  Gradient methods by condition:")
    for c in conditions:
        gm = CONDITION_GRAD_METHOD.get(c, "?")
        shots = f" | {QPU_SHOTS} shots" if gm == "spsa" else ""
        print(f"    {CONDITION_LABELS[c]:<30}  {gm}{shots}")
    print(f"\n  ⚠ = count of collapsed folds (F1 < {COLLAPSE_F1_THRESHOLD}). "
          f"Included in mean±std. Discuss in paper.")
    print()


if __name__ == "__main__":
    main()