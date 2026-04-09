# qgan/train_journal.py
# Journal Extension — 4-Backend LOOCV Training
#
# ── HOW TO RUN ───────────────────────────────────────────────────────────────
#
# LOCAL SMOKE TEST (verify code works, ~10 min):
#   python -m qgan.train_journal --mode local
#   3 subjects | 3 folds | 3 epochs | features=4 only
#   Results are IGNORED by the checkpoint system when you later run --mode full.
#
# LOCAL CPU PREVIEW (overnight, optional):
#   python -m qgan.train_journal --mode full --conditions cpu
#   3 subjects | 3 folds | CPU_EPOCHS (default 10) | features 2,3,4
#   Use to preview Conditions 1+2 before ARC. Not published results.
#
# ARC — CPU CONDITIONS (Conditions 1+2, 7 subjects, 50 epochs):
#   CPU_EPOCHS=50 python -m qgan.train_journal --mode full --conditions cpu
#
# ARC — QPU CONDITIONS (Conditions 3+4, 7 subjects, 100 epochs):
#   QPU_EPOCHS=100 python -m qgan.train_journal --mode full --conditions qpu
#
# ARC — ALL CONDITIONS IN ONE JOB:
#   CPU_EPOCHS=50 QPU_EPOCHS=100 python -m qgan.train_journal --mode full --conditions all
#
# ── OUTPUT FILES ─────────────────────────────────────────────────────────────
#   results_journal_simulator.json
#   results_journal_simulator_datanoise.json
#   results_journal_qpu_noiseless.json
#   results_journal_qpu_noise_zne.json
#   results_journal_SUMMARY.json     <- combined summary, used for paper tables
#
# ── CHECKPOINT SAFETY ────────────────────────────────────────────────────────
#   Every fold is saved immediately after completion.
#   On restart, each saved fold is validated by checking n_epochs_trained.
#   Any fold trained for fewer epochs than the current requirement is retrained.
#   This means you can safely kill and resume at any point.
#
# ── CLOPS NOTE (for paper) ───────────────────────────────────────────────────
#   shots=1    →  1 CLOP  — max noise, unsuitable for training (expectation
#                           values collapse to binary 0/1, no gradient signal)
#   shots=128  → ~100 CLOPs — IBM standard benchmark, used in all training runs
#   shots=1024 → ~1000 CLOPs — higher accuracy, ~8x slower than 128 shots
#   collect_hardware_metrics.py measures elapsed time at all three shot counts.

import argparse
import copy
import json
import os
import time
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
)

# ================================================================
#  EXPERIMENT CONFIG
# ================================================================
N_QUBITS    = 6       # Arch C: always 6 qubits
LAMBDA_GP   = 10      # WGAN-GP gradient penalty coefficient
NOISE_LEVEL = 0.1     # Gaussian sigma for Simulator+DataNoise condition

LR_G = LEARNING_RATE
LR_D = LEARNING_RATE * 5.0

# Checkpoint validation — folds saved with fewer than this many epochs
# will be detected and retrained. This is the correct validation approach
# because timing varies by machine; epoch count does not.
MIN_EPOCHS_TO_ACCEPT = 10   # any fold with n_epochs_trained < this is retrained

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
#  WGAN-GP GRADIENT PENALTY
# ================================================================
def gradient_penalty(disc, real, fake):
    bs    = real.size(0)
    alpha = torch.rand(bs, 1).float().expand_as(real)
    interp = (alpha * real.float() +
              (1 - alpha) * fake.float()).requires_grad_(True)
    d_out = disc(interp)
    grads = torch.autograd.grad(
        outputs=d_out, inputs=interp,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True, retain_graph=True
    )[0]
    return ((grads.view(bs, -1).norm(2, dim=1) - 1) ** 2).mean()


# ================================================================
#  METRICS
# ================================================================
def compute_mae(gen, data_tensor, n_features):
    """
    Mean and Std MAE between real and generated feature distributions.
    Measures how closely the generator matches the real data statistics.
    Lower std_MAE = better distribution matching.
    """
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


def compute_clf(gen, disc, data_tensor, n_features):
    """
    Classification metrics using discriminator scores.
    Threshold = 0.0 (WGAN-GP: positive score = real, negative = fake).
    Reports Accuracy, Precision, Sensitivity, Specificity, F1.
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
#  AGGREGATE LOOCV RESULTS — mean +/- std across folds
# ================================================================
def aggregate_folds(fold_results: list) -> dict:
    """
    Compute mean +/- std for all metrics across LOOCV folds.
    This is what gets reported in the journal paper tables.
    """
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
#  CHECKPOINT VALIDATION — epoch-count based (correct approach)
#
#  WHY EPOCH COUNT NOT TIMING:
#  Timing varies by machine — a fold that takes 52s/epoch on a laptop
#  may take 400s/epoch on ARC with 7 subjects. Epoch count is invariant.
#  We save n_epochs_trained in every fold result so this check is reliable.
#
#  WHY MIN_EPOCHS_TO_ACCEPT = 10:
#  Local smoke tests run 3 epochs. Full ARC runs use 50+.
#  Any fold with < 10 epochs is definitely a smoke test result.
#  This threshold safely rejects smoke test data while accepting
#  any intentional full run regardless of machine speed.
# ================================================================
def validate_saved_folds(saved_folds: list, n_epochs_required: int,
                          feat_key: str) -> list:
    """
    Accept a saved fold only if it was trained for enough epochs.
    Rejects smoke-test results automatically without timing heuristics.
    """
    valid   = []
    invalid = []

    for fold in saved_folds:
        n_trained = fold.get("history", {}).get("n_epochs_trained", None)

        # Legacy folds (before this fix) have no n_epochs_trained field.
        # Fall back to timing heuristic only for these old folds.
        if n_trained is None:
            avg_time = fold.get("history", {}).get("avg_time_per_epoch", 0)
            if avg_time >= 80:
                valid.append(fold)
            else:
                print(f"  WARNING: {feat_key} fold {fold['fold_idx']} — "
                      f"legacy checkpoint, avg_time={avg_time:.1f}s/epoch "
                      f"(< 80s threshold) — will retrain.")
                invalid.append(fold)
        else:
            if n_trained >= MIN_EPOCHS_TO_ACCEPT:
                valid.append(fold)
            else:
                print(f"  WARNING: {feat_key} fold {fold['fold_idx']} — "
                      f"only {n_trained} epochs saved "
                      f"(need >= {MIN_EPOCHS_TO_ACCEPT}) — will retrain.")
                invalid.append(fold)

    if invalid:
        print(f"  INFO: {len(invalid)} fold(s) rejected — "
              f"will be retrained from scratch.")
    return valid


# ================================================================
#  TRAINING LOOP — one condition, one LOOCV fold
# ================================================================
def train_one_fold(gen, disc, train_loader, test_data,
                   n_features, n_epochs, label):
    """
    Train WGAN-GP for one LOOCV fold.

    Generator update: 2 steps per discriminator step for stability.
    Discriminator update: 1 step with gradient penalty (lambda=10).
    Best model weights saved whenever discriminator loss improves.

    Returns (history, gen, disc) with best weights loaded.
    """
    opt_g = torch.optim.Adam(gen.parameters(),  lr=LR_G, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR_D, betas=(0.0, 0.9))

    history = {
        "gen_loss":  [],
        "disc_loss": [],
        "mean_MAE":  [],
        "std_MAE":   [],
        "mae_epochs":[],
        "times":     [],
    }
    best_disc_loss  = float("inf")
    best_gen_state  = copy.deepcopy(gen.state_dict())
    best_disc_state = copy.deepcopy(disc.state_dict())

    for epoch in range(n_epochs):
        t0 = time.time()
        g_losses, d_losses = [], []

        for (batch,) in train_loader:
            real = batch.float()
            bs   = real.shape[0]

            # Pad real features to N_QUBITS for discriminator input
            if real.shape[-1] < N_QUBITS:
                pad     = torch.zeros(bs, N_QUBITS - real.shape[-1])
                real_in = torch.cat([real, pad], dim=-1)
            else:
                real_in = real

            # Generator: 2 update steps
            for _ in range(2):
                z      = torch.randn(bs, n_features)
                fake   = gen(z)
                g_loss = -disc(fake).mean()
                opt_g.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP)
                opt_g.step()
                g_losses.append(g_loss.item())

            # Discriminator: 1 step + gradient penalty
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
        elapsed = time.time() - t0

        history["gen_loss"].append(avg_g)
        history["disc_loss"].append(avg_d)
        history["times"].append(elapsed)

        # Track best model by discriminator loss
        if avg_d < best_disc_loss:
            best_disc_loss  = avg_d
            best_gen_state  = copy.deepcopy(gen.state_dict())
            best_disc_state = copy.deepcopy(disc.state_dict())

        mae = compute_mae(gen, test_data, n_features)
        history["mean_MAE"].append(mae["mean_MAE"])
        history["std_MAE"].append(mae["std_MAE"])
        history["mae_epochs"].append(epoch + 1)

        # Print at epoch 1, every 10 epochs, and final epoch
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == n_epochs:
            print(f"      Epoch [{epoch+1:3d}/{n_epochs}] "
                  f"G:{avg_g:+.4f} D:{avg_d:+.4f} "
                  f"StdMAE:{mae['std_MAE']:.4f} "
                  f"Time:{elapsed:.1f}s  [{label}]")

    gen.load_state_dict(best_gen_state)
    disc.load_state_dict(best_disc_state)
    history["avg_time_per_epoch"] = round(float(np.mean(history["times"])), 2)
    history["n_epochs_trained"]   = n_epochs   # KEY: saved for checkpoint validation
    return history, gen, disc


# ================================================================
#  RUN ONE CONDITION — all feature counts x all LOOCV folds
# ================================================================
def run_condition(condition: str,
                  feature_sweep: list,
                  n_epochs: int,
                  subject_paths: list):
    """
    Run one backend condition across all feature counts and LOOCV folds.
    Saves a checkpoint JSON after every single fold — never lose progress.
    On restart, validates each saved fold by epoch count before skipping.
    """
    label      = CONDITION_LABELS[condition]
    out_file   = OUT_FILES[condition]
    n_subjects = len(subject_paths)
    n_folds    = n_subjects

    print(f"\n  {'='*65}")
    print(f"  CONDITION: {label}")
    print(f"  Subjects: {n_subjects}  |  LOOCV Folds: {n_folds}")
    print(f"  Features: {feature_sweep}  |  Epochs: {n_epochs}")
    print(f"  Checkpoint validation: accept folds with n_epochs >= {MIN_EPOCHS_TO_ACCEPT}")
    print(f"  {'='*65}")

    # Load existing checkpoint if available
    condition_results = {}
    if os.path.exists(out_file):
        with open(out_file) as f:
            condition_results = json.load(f)
        print(f"  Loaded checkpoint: {out_file}")

    noise_level = NOISE_LEVEL if condition == CONDITION_DATA_NOISE else 0.0

    for n_features in feature_sweep:
        feat_key = f"{n_features}_features"

        if feat_key not in condition_results:
            condition_results[feat_key] = {
                "n_features": n_features,
                "condition":  condition,
                "folds":      [],
            }

        # Validate saved folds — reject any smoke-test or incomplete results
        raw_folds   = condition_results[feat_key].get("folds", [])
        valid_folds = validate_saved_folds(raw_folds, n_epochs, feat_key)
        condition_results[feat_key]["folds"] = valid_folds
        completed_folds = len(valid_folds)

        all_data = load_all_subjects(n_features=n_features,
                                     paths=subject_paths)

        for fold_idx in range(n_folds):
            # Skip folds that already have valid results
            if fold_idx < completed_folds:
                test_subj  = os.path.basename(subject_paths[fold_idx])
                n_trained  = valid_folds[fold_idx].get(
                    "history", {}).get("n_epochs_trained", "?")
                avg_t      = valid_folds[fold_idx].get(
                    "history", {}).get("avg_time_per_epoch", "?")
                print(f"  Skipping fold {fold_idx} "
                      f"(test: {test_subj}, "
                      f"n_epochs={n_trained}, "
                      f"avg_time={avg_t}s — valid)")
                continue

            test_subject = os.path.basename(subject_paths[fold_idx])
            print(f"\n  ── Features={n_features} | "
                  f"Fold {fold_idx+1}/{n_folds} "
                  f"(test: {test_subject}) ──")

            train_loader, test_data = get_loocv_loader(
                all_data, test_idx=fold_idx,
                batch_size=BATCH_SIZE,
                noise_level=noise_level,
            )

            print(f"    Train: {len(train_loader.dataset)} epochs  "
                  f"Test: {len(test_data)} epochs")

            gen, disc = build_models(condition, N_QUBITS, n_features)

            history, gen, disc = train_one_fold(
                gen, disc, train_loader, test_data,
                n_features, n_epochs,
                label=f"{label} | f={n_features} | fold={fold_idx+1}"
            )

            mae = compute_mae(gen, test_data, n_features)
            clf = compute_clf(gen, disc, test_data, n_features)

            fold_result = {
                "fold_idx":     fold_idx,
                "test_subject": test_subject,
                "mae":          mae,
                "clf":          clf,
                "history": {
                    "n_epochs_trained":   n_epochs,                          # validation key
                    "avg_time_per_epoch": history["avg_time_per_epoch"],
                    "final_gen_loss":     round(history["gen_loss"][-1], 4),
                    "final_disc_loss":    round(history["disc_loss"][-1], 4),
                },
            }

            condition_results[feat_key]["folds"].append(fold_result)
            condition_results[feat_key]["aggregated"] = aggregate_folds(
                condition_results[feat_key]["folds"]
            )

            # Save immediately — never lose a completed fold
            with open(out_file, "w") as f:
                json.dump(condition_results, f, indent=2)
            print(f"    Checkpoint saved: {out_file} "
                  f"(fold {fold_idx+1}/{n_folds} done)")
            print(f"    FOLD RESULT → "
                  f"Acc:{clf['Accuracy']} "
                  f"Spec:{clf['Specificity']} "
                  f"F1:{clf['F1']} "
                  f"StdMAE:{mae['std_MAE']}")

        # Print aggregated result across completed folds for this feature count
        agg = condition_results[feat_key].get("aggregated", {})
        if agg:
            print(f"\n  AGGREGATED ({n_features}f, {n_folds} folds):")
            for metric in ["Accuracy", "Specificity", "F1", "std_MAE"]:
                if metric in agg:
                    m = agg[metric]
                    print(f"    {metric:<20} {m['mean']:.4f} ± {m['std']:.4f}")

    print(f"\n  Condition complete: {out_file}")
    return condition_results


# ================================================================
#  MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Journal QGAN — 4-backend LOOCV training"
    )
    parser.add_argument(
        "--mode", choices=["local", "full"], default="local",
        help=(
            "local: smoke test only — 3 epochs, features=4, first 3 subjects. "
            "Results are flagged as smoke-test and will be retrained on next full run. "
            "full: proper training — reads CPU_EPOCHS / QPU_EPOCHS env vars."
        )
    )
    parser.add_argument(
        "--conditions", choices=["cpu", "qpu", "all"], default="all",
        help=(
            "cpu:  Conditions 1+2 (Simulator, DataNoise) — run locally or on ARC. "
            "qpu:  Conditions 3+4 (QPU-Sim, QPU+ZNE) — run on ARC only. "
            "all:  All 4 conditions."
        )
    )
    args = parser.parse_args()

    # ── Mode configuration ────────────────────────────────────────
    if args.mode == "local":
        # Smoke test — just verify the code runs end to end
        # These results will be automatically rejected by checkpoint
        # validation on the next --mode full run (n_epochs_trained=3 < 10)
        base_epochs   = 3
        feature_sweep = [4]
        subject_paths = SUBJECT_PATHS[:3]
        print(f"\n  MODE: LOCAL SMOKE TEST")
        print(f"  Epochs: {base_epochs} | Features: {feature_sweep} | "
              f"Subjects: {len(subject_paths)}")
        print(f"  NOTE: These results (n_epochs=3) will be auto-rejected")
        print(f"  by the checkpoint system when you run --mode full.")

    else:
        # Full training — read epoch counts from environment variables
        # This allows ARC job scripts to set epochs without changing code
        cpu_epochs = int(os.getenv("CPU_EPOCHS", "10"))
        qpu_epochs = int(os.getenv("QPU_EPOCHS", "50"))
        feature_sweep = [2, 3, 4]
        subject_paths = SUBJECT_PATHS
        print(f"\n  MODE: FULL TRAINING RUN")
        print(f"  Features: {feature_sweep} | "
              f"Subjects: {len(subject_paths)}")
        print(f"  CPU conditions: {cpu_epochs} epochs  "
              f"[set CPU_EPOCHS env to change]")
        print(f"  QPU conditions: {qpu_epochs} epochs  "
              f"[set QPU_EPOCHS env to change]")
        print(f"  ARC recommended: CPU_EPOCHS=50 QPU_EPOCHS=100")

    conditions_to_run = CONDITION_GROUPS[args.conditions]

    print(f"\n  {'='*65}")
    print(f"  JOURNAL EXPERIMENT — 4-Backend LOOCV Study")
    print(f"  Subjects     : {len(subject_paths)} | "
          f"LOOCV folds: {len(subject_paths)}")
    print(f"  Conditions   : "
          f"{[CONDITION_LABELS[c] for c in conditions_to_run]}")
    print(f"  Features     : {feature_sweep}")
    print(f"  Architecture : Arch C — 6 qubits | ring CNOT | RX->CNOT->RY")
    print(f"  Loss         : WGAN-GP (lambda={LAMBDA_GP})")
    print(f"  Checkpoint   : accept folds with n_epochs >= {MIN_EPOCHS_TO_ACCEPT}")
    print(f"  {'='*65}")

    all_condition_results = {}

    for condition in conditions_to_run:
        if args.mode == "local":
            n_epochs = base_epochs
        elif condition in [CONDITION_SIMULATOR, CONDITION_DATA_NOISE]:
            n_epochs = cpu_epochs
        else:
            n_epochs = qpu_epochs

        print(f"\n  >> Condition: '{CONDITION_LABELS[condition]}' | "
              f"Epochs: {n_epochs}")

        result = run_condition(
            condition=condition,
            feature_sweep=feature_sweep,
            n_epochs=n_epochs,
            subject_paths=subject_paths,
        )
        all_condition_results[condition] = result

    # Save combined summary for paper tables
    with open(SUMMARY_FILE, "w") as f:
        json.dump(all_condition_results, f, indent=2)
    print(f"\n  Combined summary saved: {SUMMARY_FILE}")

    _print_summary_table(all_condition_results, feature_sweep, conditions_to_run)


# ================================================================
#  SUMMARY TABLE PRINTER
# ================================================================
def _print_summary_table(all_results, feature_sweep, conditions):
    """Print paper-ready results table to console."""
    # Count folds from first available condition
    num_folds = 0
    for cond in conditions:
        if cond in all_results:
            keys = list(all_results[cond].keys())
            if keys:
                num_folds = len(all_results[cond][keys[0]].get("folds", []))
                break

    print(f"\n  {'='*90}")
    print(f"  FINAL RESULTS TABLE  "
          f"(mean ± std across {num_folds} LOOCV folds)")
    print(f"  {'─'*90}")
    print(f"  {'Condition':<28} {'Feat':<6} {'Acc':<14} {'Spec':<14} "
          f"{'F1':<14} {'StdMAE':<14} {'Time/ep'}")
    print(f"  {'─'*90}")

    for condition in conditions:
        label = CONDITION_LABELS[condition]
        if condition not in all_results:
            continue
        printed_any = False
        for n_f in feature_sweep:
            feat_key = f"{n_f}_features"
            agg = all_results[condition].get(feat_key, {}).get("aggregated", {})
            if not agg:
                continue
            printed_any = True

            def fmt(key):
                if key in agg:
                    return f"{agg[key]['mean']:.3f}±{agg[key]['std']:.3f}"
                return "---"

            print(f"  {label:<28} {n_f:<6} "
                  f"{fmt('Accuracy'):<14} "
                  f"{fmt('Specificity'):<14} "
                  f"{fmt('F1'):<14} "
                  f"{fmt('std_MAE'):<14} "
                  f"{fmt('avg_time_per_epoch')}s")
        if printed_any:
            print(f"  {'─'*90}")

    print(f"  {'='*90}\n")


if __name__ == "__main__":
    main()