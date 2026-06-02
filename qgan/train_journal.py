# qgan/train_journal.py
# Journal Extension — Multi-Baseline LOOCV Training
#
# ── WHAT CHANGED vs PREVIOUS VERSION (and WHY) ─────────────────────────────
#
#  FIX 1 (CRITICAL — downstream evaluation methodology):
#    evaluate_downstream() previously split the TEST SUBJECT's data 80/20
#    and trained the downstream classifier on 80% of the held-out subject.
#    This defeats the purpose of LOOCV — the downstream classifier was
#    trained on data it should never have seen.
#    CORRECT approach: train downstream classifier on the SAME training
#    data the GAN used (N-1 subjects), evaluate on the full test subject.
#    train_journal.py now passes train_feats + train_labels to
#    evaluate_downstream(), not test_feats.
#
#  FIX 2 (CRITICAL — scaler bias in augmented comparison):
#    Previously the StandardScaler was fit SEPARATELY on real training data
#    and on augmented training data. Fitting on augmented data (which
#    includes synthetic samples clamped to [-1,1]) changes the scale
#    statistics, making the real vs augmented comparison unfair.
#    CORRECT approach: fit scaler on REAL training data only, then
#    apply that same scaler to the augmented set and to the test set.
#
#  FIX 3 (Statistical testing minimum folds):
#    Wilcoxon test now requires >= 5 folds (was >= 3). With only 3 data
#    points the Wilcoxon test has essentially no statistical power.
#    10-subject LOOCV gives 10 folds — sufficient for p-value reporting.
#
#  FIX 4 (Parameter counts saved to results JSON — Nature requirement):
#    Fold results now include n_params_gen and n_params_disc.
#    This enables the parameter comparison table required for Nature.
#    build_models() returns these counts (see models_journal.py FIX 2).
#
#  FIX 5 (get_loocv_loader signature change):
#    data_loader_journal.py now returns (loader, train_f, train_l, test_f, test_l)
#    All call sites updated to unpack 5 values instead of 3.
#
#  FIX 6 (Dynamic minority class in downstream evaluation):
#    build_augmented_dataset() now auto-detects the minority class per fold
#    (see data_loader_journal.py FIX 2). No code change needed here —
#    just don't pass target_stage=1.
#
# ── REVIEWER 2 RESPONSES ────────────────────────────────────────────────────
#   Weakness 1 (weak baseline):  ClassicalBCEGenerator (conference, small MLP)
#                                 ClassicalWGANGenerator (fair, same loss)
#                                 DCGANStyleGenerator (modern method)
#   Weakness 2 (simplified feat): statistical (4) | spectral (5) | combined (9)
#                                 spectral = clinically meaningful AASM bands
#   Weakness 3 (no downstream):  LOOCV SVM + RandomForest, N1 minority F1,
#                                 delta vs real-only, Wilcoxon significance
#
# ── HOW TO RUN ──────────────────────────────────────────────────────────────
#   LOCAL SMOKE TEST (~10 min):
#     python -m qgan.train_journal --mode local
#
#   FULL CPU (classical + quantum simulator, all feature sets):
#     CPU_EPOCHS=50 python -m qgan.train_journal --mode full --conditions cpu
#
#   FULL QPU (hardware noise simulation):
#     QPU_EPOCHS=100 python -m qgan.train_journal --mode full --conditions qpu
#
#   ALL CONDITIONS:
#     CPU_EPOCHS=50 QPU_EPOCHS=100 python -m qgan.train_journal \
#       --mode full --conditions all
#
#   SINGLE FEATURE SET:
#     FEATURE_SET=combined python -m qgan.train_journal --mode full
#
#   RUN EXPRESSIBILITY SWEEP (once, before training):
#     python -m qgan.train_journal --expressibility-only

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
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from qgan.config import LEARNING_RATE, GRAD_CLIP, BATCH_SIZE
from qgan.data_loader_journal import (
    load_all_subjects, get_loocv_loader,
    build_augmented_dataset, SUBJECT_PATHS,
)
from qgan.models_journal import (
    build_models,
    ALL_CONDITIONS,
    CONDITION_SIMULATOR, CONDITION_DATA_NOISE,
    CONDITION_QPU_SIM,   CONDITION_QPU_ZNE,
    CONDITION_LABELS, CONDITION_GRAD_METHOD,
    GEN_CLASSICAL_BCE, GEN_CLASSICAL_WGAN,
    GEN_DCGAN, GEN_QUANTUM,
    GENERATOR_LABELS,
    run_expressibility_sweep,
)

# ================================================================
#  EXPERIMENT CONFIG
# ================================================================
N_QUBITS_FOR_FEATURES = {
    "statistical": 6,    # 4 features + 2 ancilla
    "spectral":    7,    # 5 features + 2 ancilla
    "combined":    11,   # 9 features + 2 ancilla
}
FEATURE_SET_N = {"statistical": 4, "spectral": 5, "combined": 9}
DEFAULT_FEATURE_SETS = ["statistical", "spectral", "combined"]

LAMBDA_GP   = 10
NOISE_LEVEL = 0.1

LR_G = LEARNING_RATE
LR_D = LEARNING_RATE * 5.0

MIN_EPOCHS_TO_ACCEPT  = 10
COLLAPSE_F1_THRESHOLD = 0.15
WALL_CLOCK_BUFFER_S   = 600
MIN_WILCOXON_FOLDS    = 5    # FIX 3: raised from 3 to 5

CONDITION_GROUPS = {
    "cpu": [CONDITION_SIMULATOR, CONDITION_DATA_NOISE],
    "qpu": [CONDITION_QPU_SIM,   CONDITION_QPU_ZNE],
    "all": ALL_CONDITIONS,
}

SUMMARY_FILE        = "results_journal_SUMMARY.json"
STATS_FILE          = "results_journal_STATS.json"
EXPRESSIBILITY_FILE = "expressibility.json"


# ================================================================
#  UTILITIES
# ================================================================
def _seconds_until_wall_limit() -> float:
    end = os.getenv("SLURM_JOB_END_TIME")
    if end is None:
        return float("inf")
    try:
        return float(end) - time.time()
    except ValueError:
        return float("inf")


def _fmt_seconds(s: float) -> str:
    s = max(0.0, s)
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def _atomic_write(path: str, data: dict) -> None:
    """Write JSON atomically — safe against SLURM SIGKILL mid-write."""
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def out_file_path(condition: str, feature_set: str,
                   generator_type: str = None) -> str:
    if generator_type and generator_type != GEN_QUANTUM:
        return f"results_{condition}_{feature_set}_{generator_type}.json"
    return f"results_{condition}_{feature_set}.json"


# ================================================================
#  WGAN-GP GRADIENT PENALTY
# ================================================================
def gradient_penalty(disc, real: torch.Tensor,
                      fake: torch.Tensor) -> torch.Tensor:
    bs    = real.size(0)
    alpha = torch.rand(bs, 1).expand_as(real)
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
#  GENERATIVE FIDELITY METRICS
# ================================================================
def compute_mae(gen, data_tensor: torch.Tensor,
                n_features: int) -> dict:
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


def compute_clf(gen, disc, data_tensor: torch.Tensor,
                n_features: int, disc_input_dim: int) -> dict:
    """
    Real/fake discrimination metrics.
    Threshold = 0.0 (WGAN-GP convention: positive=real, negative=fake).
    """
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, confusion_matrix)
    gen.eval(); disc.eval()
    n = min(len(data_tensor), 100)
    with torch.no_grad():
        real = data_tensor[:n, :n_features].float()
        z    = torch.randn(n, n_features)
        fake = gen(z)

        if real.shape[-1] < disc_input_dim:
            pad     = torch.zeros(n, disc_input_dim - real.shape[-1])
            real_in = torch.cat([real, pad], dim=-1)
        else:
            real_in = real[:, :disc_input_dim]

        fake_in = fake[:, :disc_input_dim]

        rs = disc(real_in).squeeze().detach().numpy()
        fs = disc(fake_in).squeeze().detach().numpy()

    scores = np.concatenate([rs, fs])
    labels = np.array([1]*n + [0]*n)
    preds  = (scores > 0.0).astype(int)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    gen.train(); disc.train()

    return {
        "Accuracy":    round(accuracy_score(labels, preds), 4),
        "Precision":   round(precision_score(labels, preds, zero_division=0), 4),
        "Sensitivity": round(recall_score(labels, preds, zero_division=0), 4),
        "Specificity": round(float(tn/(tn+fp)) if (tn+fp) > 0 else 0.0, 4),
        "F1":          round(f1_score(labels, preds, zero_division=0), 4),
    }


# ================================================================
#  DOWNSTREAM CLASSIFIER EVALUATION (FIX 1 + FIX 2)
#
#  FIX 1: NOW takes train_feats/train_labels (the GAN's training data),
#          trains downstream classifiers on that + synthetic samples,
#          then evaluates on the HELD-OUT test subject.
#          Previously it was splitting the test subject data — WRONG.
#
#  FIX 2: StandardScaler is fit on REAL training data only,
#          then applied to both real and augmented training sets.
#          Previously scaler was re-fit on augmented data — unfair.
# ================================================================
def evaluate_downstream(train_feats: torch.Tensor,
                          train_labels: torch.Tensor,
                          test_feats: torch.Tensor,
                          test_labels: torch.Tensor,
                          generator,
                          n_features: int) -> dict:
    """
    Train SVM and RandomForest on:
      (a) real training data only (baseline)
      (b) real training data + synthetic minority-class samples (augmented)
    Evaluate on the held-out test subject.

    FIX 1: train_feats = training subjects (N-1 subjects), NOT test subject.
    FIX 2: scaler fit on real train data only, applied uniformly.

    Returns dict with keys: svm_real, svm_aug, rf_real, rf_aug,
    each containing: Accuracy, MacroF1, N1_F1, per_class_F1,
                     delta_MacroF1, delta_N1_F1, n_train, n_test, n_synthetic
    """
    X_train_real = train_feats[:, :n_features].numpy()
    y_train_real = train_labels.numpy()
    X_test       = test_feats[:, :n_features].numpy()
    y_test       = test_labels.numpy()

    # Build augmented training set (FIX 6: minority class auto-detected)
    aug_feats, aug_labels = build_augmented_dataset(
        train_feats, train_labels, generator
        # target_stage=None → auto-detects minority class per fold
    )
    X_train_aug = aug_feats[:, :n_features].numpy()
    y_train_aug = aug_labels.numpy()

    # FIX 2: fit scaler on REAL training data only
    scaler = StandardScaler()
    scaler.fit(X_train_real)

    X_train_real_sc = scaler.transform(X_train_real)
    X_train_aug_sc  = scaler.transform(X_train_aug)   # same scaler!
    X_test_sc       = scaler.transform(X_test)

    results       = {}
    unique_classes = np.unique(np.concatenate([y_train_real, y_test]))
    stage_names    = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    for clf_name, clf_real_cls, clf_aug_cls in [
        ("svm",
         SVC(kernel="rbf", class_weight="balanced", random_state=42),
         SVC(kernel="rbf", class_weight="balanced", random_state=42)),
        ("rf",
         RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                random_state=42, n_jobs=-1),
         RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                random_state=42, n_jobs=-1)),
    ]:
        for tag, clf, X_tr, y_tr in [
            ("real", clf_real_cls, X_train_real_sc, y_train_real),
            ("aug",  clf_aug_cls,  X_train_aug_sc,  y_train_aug),
        ]:
            key = f"{clf_name}_{tag}"
            try:
                clf.fit(X_tr, y_tr)
                preds    = clf.predict(X_test_sc)
                macro_f1 = f1_score(y_test, preds, average="macro",
                                    zero_division=0)
                n1_f1    = f1_score(y_test, preds, labels=[1],
                                    average="micro", zero_division=0)

                report = classification_report(
                    y_test, preds,
                    labels=list(unique_classes),
                    output_dict=True,
                    zero_division=0,
                )
                per_class = {}
                for cls_int in unique_classes:
                    cls_str = str(int(cls_int))
                    if cls_str in report:
                        per_class[stage_names.get(int(cls_int), cls_str)] = {
                            "F1":        round(report[cls_str]["f1-score"], 4),
                            "Precision": round(report[cls_str]["precision"], 4),
                            "Recall":    round(report[cls_str]["recall"], 4),
                        }

                n_synthetic = int(len(y_train_aug) - len(y_train_real)) \
                              if tag == "aug" else 0

                results[key] = {
                    "Accuracy":     round(accuracy_score(y_test, preds), 4),
                    "MacroF1":      round(macro_f1, 4),
                    "N1_F1":        round(n1_f1, 4),
                    "per_class_F1": per_class,
                    "n_train":      int(len(y_tr)),
                    "n_test":       int(len(y_test)),
                    "n_synthetic":  n_synthetic,
                }
            except Exception as e:
                results[key] = {"error": str(e)}

    # Compute deltas (augmented vs real-only)
    for clf_name in ["svm", "rf"]:
        rk = f"{clf_name}_real"
        ak = f"{clf_name}_aug"
        if (rk in results and ak in results and
                "MacroF1" in results[rk] and "MacroF1" in results[ak]):
            results[ak]["delta_MacroF1"] = round(
                results[ak]["MacroF1"] - results[rk]["MacroF1"], 4)
            results[ak]["delta_N1_F1"] = round(
                results[ak]["N1_F1"] - results[rk]["N1_F1"], 4)

    return results


# ================================================================
#  CHECKPOINT VALIDATION
# ================================================================
def get_valid_fold_indices(saved_folds: list) -> set:
    valid = set()
    for fold in saved_folds:
        idx       = fold.get("fold_idx")
        n_trained = fold.get("history", {}).get("n_epochs_trained")
        if idx is None:
            continue
        if n_trained is None:
            avg_t = fold.get("history", {}).get("avg_time_per_epoch", 0)
            if avg_t >= 80:
                valid.add(idx)
        elif n_trained >= MIN_EPOCHS_TO_ACCEPT:
            valid.add(idx)
    return valid


# ================================================================
#  AGGREGATE LOOCV RESULTS
# ================================================================
def aggregate_folds(fold_results: list) -> dict:
    scalar_keys = [
        "mean_MAE", "std_MAE",
        "Accuracy", "Precision", "Sensitivity", "Specificity", "F1",
        "avg_time_per_epoch",
    ]
    downstream_keys = [
        ("svm_aug", "MacroF1"),        ("svm_aug", "N1_F1"),
        ("svm_aug", "delta_MacroF1"),  ("svm_aug", "delta_N1_F1"),
        ("rf_aug",  "MacroF1"),        ("rf_aug",  "N1_F1"),
        ("rf_aug",  "delta_MacroF1"),  ("rf_aug",  "delta_N1_F1"),
        ("svm_real","MacroF1"),        ("rf_real", "MacroF1"),
    ]
    agg = {}

    for k in scalar_keys:
        vals = []
        for r in fold_results:
            for src in (r.get("mae", {}), r.get("clf", {}),
                        r.get("history", {})):
                if k in src:
                    vals.append(src[k])
                    break
        if vals:
            agg[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)),  4),
                "all":  [round(v, 4) for v in vals],
            }

    for clf_tag, metric in downstream_keys:
        vals = []
        for r in fold_results:
            ds = r.get("downstream", {})
            v  = ds.get(clf_tag, {}).get(metric)
            if v is not None:
                vals.append(v)
        if vals:
            key = f"downstream_{clf_tag}_{metric}"
            agg[key] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)),  4),
                "all":  [round(v, 4) for v in vals],
            }

    # Parameter summary — take from first fold
    for fold in fold_results:
        if "n_params_gen" in fold:
            agg["n_params_gen"]  = fold["n_params_gen"]
            agg["n_params_disc"] = fold["n_params_disc"]
            break

    return agg


# ================================================================
#  STATISTICAL SIGNIFICANCE (FIX 3: min 5 folds)
# ================================================================
def compute_significance_tests(all_results: dict) -> dict:
    """
    Pairwise Wilcoxon signed-rank test across LOOCV folds.
    FIX 3: requires >= MIN_WILCOXON_FOLDS (5) folds.
    Compares QWGAN-GP (Simulator) vs each classical baseline.
    Primary metrics: StdMAE and SVM-aug MacroF1.
    """
    stats = {}

    for feature_set in DEFAULT_FEATURE_SETS:
        n_feat   = FEATURE_SET_N[feature_set]
        feat_key = f"{feature_set}_{n_feat}feat"
        stats[feature_set] = {}

        # Get QGAN baseline fold values
        qgan_std_mae  = None
        qgan_macro_f1 = None
        for key, res in all_results.items():
            if CONDITION_SIMULATOR in key and GEN_QUANTUM in key:
                cdata = res.get(feat_key, {})
                agg   = cdata.get("aggregated", {})
                if "std_MAE" in agg:
                    qgan_std_mae  = agg["std_MAE"].get("all", [])
                k = "downstream_svm_aug_MacroF1"
                if k in agg:
                    qgan_macro_f1 = agg[k].get("all", [])
                break

        if qgan_std_mae is None or len(qgan_std_mae) < MIN_WILCOXON_FOLDS:
            stats[feature_set]["_note"] = (
                f"QGAN folds < {MIN_WILCOXON_FOLDS} — "
                f"Wilcoxon test not computed (insufficient power)."
            )
            continue

        for gen_type in [GEN_CLASSICAL_WGAN, GEN_DCGAN, GEN_CLASSICAL_BCE]:
            gen_label  = GENERATOR_LABELS.get(gen_type, gen_type)
            test_entry = {}
            baseline_vals = None

            for key, res in all_results.items():
                if gen_type in key and feature_set in key:
                    cdata = res.get(feat_key, {})
                    agg   = cdata.get("aggregated", {})
                    if "std_MAE" in agg:
                        baseline_vals = agg["std_MAE"].get("all", [])
                    break

            if baseline_vals is None or len(baseline_vals) < MIN_WILCOXON_FOLDS:
                test_entry["StdMAE_wilcoxon"] = {
                    "error": f"Baseline folds < {MIN_WILCOXON_FOLDS}"
                }
            else:
                n = min(len(qgan_std_mae), len(baseline_vals))
                try:
                    stat, p = wilcoxon(qgan_std_mae[:n], baseline_vals[:n],
                                       alternative="less")
                    test_entry["StdMAE_wilcoxon"] = {
                        "statistic":       round(float(stat), 4),
                        "p_value":         round(float(p), 4),
                        "significant_p05": bool(p < 0.05),
                        "n_folds":         n,
                        "note":            "H1: QGAN StdMAE < Classical StdMAE",
                    }
                except Exception as e:
                    test_entry["StdMAE_wilcoxon"] = {"error": str(e)}

            stats[feature_set][gen_label] = test_entry

    return stats


# ================================================================
#  TRAINING LOOP — one LOOCV fold
# ================================================================
def train_one_fold(gen, disc, train_loader,
                   eval_feats: torch.Tensor,
                   n_features: int, disc_input_dim: int,
                   n_epochs: int, label: str,
                   use_bce: bool = False) -> tuple:
    """
    WGAN-GP (or BCE) training for one LOOCV fold.
    Generator updated 2x per discriminator update.

    eval_feats: used for MAE/CLF metrics only (test subject).
    """
    opt_g = torch.optim.Adam(gen.parameters(),  lr=LR_G, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR_D, betas=(0.0, 0.9))

    if use_bce:
        bce_loss = torch.nn.BCEWithLogitsLoss()

    history    = {"gen_loss": [], "disc_loss": [], "times": []}
    best_d_loss = float("inf")
    best_gen    = copy.deepcopy(gen.state_dict())
    best_disc   = copy.deepcopy(disc.state_dict())
    fold_start  = time.time()

    for epoch in range(n_epochs):
        t0 = time.time()
        g_losses, d_losses = [], []

        for (batch,) in train_loader:
            real = batch.float()
            bs   = real.shape[0]

            if real.shape[-1] < disc_input_dim:
                pad     = torch.zeros(bs, disc_input_dim - real.shape[-1])
                real_in = torch.cat([real, pad], dim=-1)
            else:
                real_in = real[:, :disc_input_dim]

            # Generator update ×2
            for _ in range(2):
                z    = torch.randn(bs, n_features)
                fake = gen(z)
                fd   = fake[:, :disc_input_dim]

                if use_bce:
                    rl     = torch.ones(bs, 1)
                    g_loss = bce_loss(disc(fd), rl)
                else:
                    g_loss = -disc(fd).mean()

                opt_g.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP)
                opt_g.step()
                g_losses.append(g_loss.item())

            # Discriminator update ×1
            z    = torch.randn(bs, n_features)
            fake = gen(z).detach()
            fd   = fake[:, :disc_input_dim]

            if use_bce:
                rl     = torch.ones(bs, 1)
                fl     = torch.zeros(bs, 1)
                d_loss = (bce_loss(disc(real_in), rl) +
                          bce_loss(disc(fd), fl)) / 2
            else:
                gp     = gradient_penalty(disc, real_in, fd)
                d_loss = (-disc(real_in).mean() +
                           disc(fd).mean() +
                           LAMBDA_GP * gp)

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

        if avg_d < best_d_loss:
            best_d_loss = avg_d
            best_gen    = copy.deepcopy(gen.state_dict())
            best_disc   = copy.deepcopy(disc.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == n_epochs:
            done = time.time() - fold_start
            eta  = _fmt_seconds((done / (epoch+1)) * (n_epochs - epoch - 1))
            print(f"      Epoch [{epoch+1:3d}/{n_epochs}] "
                  f"G:{avg_g:+.4f}  D:{avg_d:+.4f}  "
                  f"t:{elapsed:.1f}s  ETA:{eta}  [{label}]")

    gen.load_state_dict(best_gen)
    disc.load_state_dict(best_disc)

    history["avg_time_per_epoch"] = round(float(np.mean(history["times"])), 2)
    history["total_fold_time_s"]  = round(time.time() - fold_start, 1)
    history["n_epochs_trained"]   = n_epochs

    mae = compute_mae(gen, eval_feats, n_features)
    clf = compute_clf(gen, disc, eval_feats, n_features, disc_input_dim)

    return history, mae, clf, gen, disc


# ================================================================
#  RUN ONE CONDITION × GENERATOR TYPE
# ================================================================
def run_one_config(condition: str,
                   generator_type: str,
                   feature_set: str,
                   n_epochs: int,
                   subject_paths: list,
                   all_features: list,
                   all_labels: list) -> dict:
    """
    Run LOOCV for one (condition, generator_type, feature_set) combination.
    Checkpointed atomically after each fold.
    Returns results dict.
    """
    is_quantum = (generator_type == GEN_QUANTUM)
    is_bce     = (generator_type == GEN_CLASSICAL_BCE)
    label      = (CONDITION_LABELS.get(condition, condition)
                  if is_quantum
                  else GENERATOR_LABELS.get(generator_type, generator_type))
    out_file   = out_file_path(condition, feature_set, generator_type)
    n_folds    = len(subject_paths)
    n_features = FEATURE_SET_N[feature_set]
    n_qubits   = N_QUBITS_FOR_FEATURES[feature_set]
    feat_key   = f"{feature_set}_{n_features}feat"

    # Load checkpoint
    results: dict = {}
    if os.path.exists(out_file):
        try:
            with open(out_file) as f:
                results = json.load(f)
            print(f"  Loaded checkpoint: {out_file}")
        except json.JSONDecodeError:
            print(f"  WARNING: {out_file} corrupt — starting fresh.")

    if feat_key not in results:
        results[feat_key] = {
            "n_features":     n_features,
            "feature_set":    feature_set,
            "condition":      condition,
            "generator_type": generator_type,
            "folds":          [],
        }

    raw_folds     = results[feat_key].get("folds", [])
    saved_by_idx  = {f["fold_idx"]: f for f in raw_folds if "fold_idx" in f}
    valid_indices = get_valid_fold_indices(raw_folds)

    results[feat_key]["folds"] = [
        saved_by_idx[i] for i in sorted(valid_indices) if i in saved_by_idx
    ]

    pending = [i for i in range(n_folds) if i not in valid_indices]
    if not pending:
        print(f"  All folds done for {feat_key}. Skipping.")
        return results

    print(f"\n  {'─'*60}")
    print(f"  Config: {label} | {feature_set} ({n_features} features, "
          f"{n_qubits} qubits)")
    print(f"  Folds pending: {pending}")

    noise_level = NOISE_LEVEL if condition == CONDITION_DATA_NOISE else 0.0
    disc_input_dim = n_qubits if is_quantum else n_features

    for fold_idx in range(n_folds):
        if fold_idx in valid_indices:
            continue
        if _seconds_until_wall_limit() < WALL_CLOCK_BUFFER_S:
            print(f"\n  WALL-CLOCK GUARD: stopping cleanly.")
            return results

        test_subj = os.path.basename(subject_paths[fold_idx])
        print(f"\n  ── Fold {fold_idx+1}/{n_folds} (test: {test_subj}) ──")

        # FIX 5: unpack 5 values (added train_f, train_l)
        train_loader, train_f, train_l, test_f, test_l = get_loocv_loader(
            all_features, all_labels,
            test_idx=fold_idx,
            batch_size=BATCH_SIZE,
            noise_level=noise_level,
            do_oversample=True,
        )

        # FIX 4: build_models now returns param counts
        gen, disc, n_params_gen, n_params_disc = build_models(
            condition=condition,
            n_qubits=n_qubits,
            n_features=n_features,
            generator_type=generator_type,
        )

        fold_label = f"{label} | {feature_set} | fold {fold_idx+1}"
        history, mae, clf, gen, disc = train_one_fold(
            gen, disc, train_loader,
            eval_feats=test_f,
            n_features=n_features,
            disc_input_dim=disc_input_dim,
            n_epochs=n_epochs,
            label=fold_label,
            use_bce=is_bce,
        )

        # FIX 1 + FIX 2: pass train_f/train_l, not test_f
        try:
            downstream = evaluate_downstream(
                train_feats=train_f,
                train_labels=train_l,
                test_feats=test_f,
                test_labels=test_l,
                generator=gen,
                n_features=n_features,
            )
        except Exception as e:
            downstream = {"error": str(e)}
            warnings.warn(f"Downstream evaluation failed (fold {fold_idx}): {e}")

        is_collapsed = clf["F1"] < COLLAPSE_F1_THRESHOLD
        if is_collapsed:
            print(f"    WARNING: Fold {fold_idx} collapsed (F1={clf['F1']:.4f}).")

        fold_result = {
            "fold_idx":       fold_idx,
            "test_subject":   test_subj,
            "collapsed":      is_collapsed,
            "mae":            mae,
            "clf":            clf,
            "downstream":     downstream,
            "n_params_gen":   n_params_gen,    # FIX 4
            "n_params_disc":  n_params_disc,   # FIX 4
            "history": {
                "n_epochs_trained":   n_epochs,
                "avg_time_per_epoch": history["avg_time_per_epoch"],
                "total_fold_time_s":  history["total_fold_time_s"],
                "final_gen_loss":     round(history["gen_loss"][-1], 4),
                "final_disc_loss":    round(history["disc_loss"][-1], 4),
            },
        }

        results[feat_key]["folds"].append(fold_result)
        results[feat_key]["folds"].sort(key=lambda f: f["fold_idx"])
        results[feat_key]["aggregated"] = aggregate_folds(
            results[feat_key]["folds"]
        )

        _atomic_write(out_file, results)
        valid_indices.add(fold_idx)

        svm_n1 = (downstream.get("svm_aug", {}).get("N1_F1", "N/A")
                  if "error" not in downstream else "ERR")
        delta  = (downstream.get("svm_aug", {}).get("delta_MacroF1", "N/A")
                  if "error" not in downstream else "ERR")
        print(f"    FOLD RESULT → "
              f"Acc:{clf['Accuracy']:.4f}  F1:{clf['F1']:.4f}  "
              f"StdMAE:{mae['std_MAE']:.4f}  "
              f"SVM_N1_F1:{svm_n1}  ΔMacroF1:{delta}  "
              f"GenParams:{n_params_gen}  "
              f"Time:{_fmt_seconds(history['total_fold_time_s'])}")

    print(f"\n  Checkpoint saved: {out_file}")
    return results


# ================================================================
#  MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Journal QGAN — Multi-baseline LOOCV"
    )
    parser.add_argument("--mode",       choices=["local", "full"], default="local")
    parser.add_argument("--conditions", choices=["cpu", "qpu", "all"], default="cpu")
    parser.add_argument("--expressibility-only", action="store_true",
                        help="Run only expressibility sweep then exit.")
    args = parser.parse_args()

    print(f"\n  {'='*70}")
    print(f"  JOURNAL QGAN — MULTI-BASELINE LOOCV")
    print(f"  Python: {sys.version.split()[0]}")
    try:
        import pennylane as qml
        print(f"  PennyLane: {qml.__version__}")
    except ImportError:
        print(f"  PennyLane: NOT FOUND")
    print(f"  SLURM job: {os.getenv('SLURM_JOB_ID', 'not set')}")

    # Expressibility-only mode
    if args.expressibility_only:
        print(f"\n  Running expressibility sweep only...")
        run_expressibility_sweep(output_file=EXPRESSIBILITY_FILE)
        print(f"  Done. Saved: {EXPRESSIBILITY_FILE}")
        return

    feature_set_env = os.getenv("FEATURE_SET", "all")
    feature_sets    = DEFAULT_FEATURE_SETS if feature_set_env == "all" \
                      else [feature_set_env]

    if args.mode == "local":
        n_epochs_cpu  = 3
        n_epochs_qpu  = 3
        subject_paths = SUBJECT_PATHS[:3]
        feature_sets  = ["statistical"]
        print(f"\n  MODE: LOCAL SMOKE TEST (3 epochs, 3 subjects, statistical)")
    else:
        n_epochs_cpu  = int(os.getenv("CPU_EPOCHS", "50"))
        n_epochs_qpu  = int(os.getenv("QPU_EPOCHS", "100"))
        subject_paths = [p for p in subject_paths if os.path.exists(p)]
        print(f"\n  MODE: FULL | CPU_EPOCHS={n_epochs_cpu} "
              f"QPU_EPOCHS={n_epochs_qpu}")
        print(f"  Feature sets: {feature_sets}")

    conditions_to_run = CONDITION_GROUPS[args.conditions]
    print(f"  Conditions: "
          f"{[CONDITION_LABELS.get(c, c) for c in conditions_to_run]}")
    print(f"  Subjects: {len(subject_paths)}")
    print(f"  {'='*70}")

    all_summary = {}

    for feature_set in feature_sets:
        n_features = FEATURE_SET_N[feature_set]
        print(f"\n  {'#'*70}")
        print(f"  FEATURE SET: {feature_set} ({n_features} features)")
        print(f"  {'#'*70}")

        # load_all_subjects returns (features, labels, scaler) — FIX 1 in data loader
        result = load_all_subjects(
            n_features=n_features,
            feature_set=feature_set,
            paths=subject_paths,
        )
        all_features, all_labels = result[0], result[1]
        scaler_info = result[2] if len(result) > 2 else {}

        if len(all_features) < 2:
            print(f"  Skipping {feature_set}: need >= 2 subjects.")
            continue

        # Save scaler info for reproducibility
        if scaler_info:
            _atomic_write(f"scaler_{feature_set}.json", scaler_info)

        # 1. Classical baselines
        if args.conditions in ("cpu", "all"):
            for gen_type in [GEN_CLASSICAL_BCE,
                             GEN_CLASSICAL_WGAN,
                             GEN_DCGAN]:
                print(f"\n  Running classical: "
                      f"{GENERATOR_LABELS[gen_type]}")
                res = run_one_config(
                    condition=CONDITION_SIMULATOR,
                    generator_type=gen_type,
                    feature_set=feature_set,
                    n_epochs=n_epochs_cpu,
                    subject_paths=subject_paths,
                    all_features=all_features,
                    all_labels=all_labels,
                )
                key = f"classical_{gen_type}_{feature_set}"
                all_summary[key] = res

        # 2. Quantum conditions
        for condition in conditions_to_run:
            n_epochs = (n_epochs_qpu
                        if condition in (CONDITION_QPU_SIM, CONDITION_QPU_ZNE)
                        else n_epochs_cpu)
            print(f"\n  Running quantum: {CONDITION_LABELS[condition]}")
            res = run_one_config(
                condition=condition,
                generator_type=GEN_QUANTUM,
                feature_set=feature_set,
                n_epochs=n_epochs,
                subject_paths=subject_paths,
                all_features=all_features,
                all_labels=all_labels,
            )
            key = f"{condition}_{feature_set}"
            all_summary[key] = res

    _atomic_write(SUMMARY_FILE, all_summary)
    print(f"\n  Summary saved: {SUMMARY_FILE}")

    try:
        stats = compute_significance_tests(all_summary)
        _atomic_write(STATS_FILE, stats)
        print(f"  Significance tests saved: {STATS_FILE}")
        _print_stats_summary(stats)
    except Exception as e:
        warnings.warn(f"Significance testing failed: {e}")

    _print_summary_table(all_summary, feature_sets)


# ================================================================
#  PRINTING
# ================================================================
def _print_summary_table(all_results: dict, feature_sets: list) -> None:
    W = 120
    print(f"\n  {'='*W}")
    print(f"  FINAL RESULTS (mean ± std across LOOCV folds)")
    print(f"  {'─'*W}")
    hdr = (f"  {'Config':<42} {'Feat':<12} {'Acc':<14} {'F1':<14} "
           f"{'StdMAE':<14} {'SVM_N1_F1':<14} {'Params_G':<12} {'Time/ep'}")
    print(hdr)
    print(f"  {'─'*W}")

    for key, res in all_results.items():
        for feat_key, cdata in res.items():
            if not isinstance(cdata, dict) or "aggregated" not in cdata:
                continue
            agg  = cdata["aggregated"]
            fs   = cdata.get("feature_set", "?")
            nf   = cdata.get("n_features", "?")
            gt   = cdata.get("generator_type", GEN_QUANTUM)
            cond = cdata.get("condition", "?")
            cfg  = (GENERATOR_LABELS.get(gt, gt)
                    if gt != GEN_QUANTUM
                    else CONDITION_LABELS.get(cond, cond))
            params_g = agg.get("n_params_gen", "?")

            def fmt(k):
                return (f"{agg[k]['mean']:.3f}±{agg[k]['std']:.3f}"
                        if k in agg else "---")

            print(f"  {cfg:<42} {fs}/{nf}f{'':<2} "
                  f"{fmt('Accuracy'):<14} "
                  f"{fmt('F1'):<14} "
                  f"{fmt('std_MAE'):<14} "
                  f"{fmt('downstream_svm_aug_N1_F1'):<14} "
                  f"{str(params_g):<12} "
                  f"{fmt('avg_time_per_epoch')}s")

    print(f"  {'='*W}\n")


def _print_stats_summary(stats: dict) -> None:
    print(f"\n  Wilcoxon Signed-Rank (H1: QGAN StdMAE < Classical):")
    for feat_set, comparisons in stats.items():
        if "_note" in comparisons:
            print(f"  {feat_set}: {comparisons['_note']}")
            continue
        print(f"  Feature set: {feat_set}")
        for baseline, tests in comparisons.items():
            for metric, result in tests.items():
                if isinstance(result, dict) and "error" not in result:
                    sig = ("** SIGNIFICANT **"
                           if result.get("significant_p05") else "n.s.")
                    print(f"    vs {baseline:<35} {metric:<22} "
                          f"p={result['p_value']:.4f} n={result['n_folds']} "
                          f"[{sig}]")


if __name__ == "__main__":
    main()