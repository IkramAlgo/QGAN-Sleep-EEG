# qgan/evaluate_augmentation.py
#
# Downstream Classifier Evaluation — Addresses Reviewer 2 Concern 3
#
# "The evaluation does not clearly demonstrate improvement on the
#  actual downstream task (sleep stage classification)"
#
# WHAT THIS DOES:
#   1. Loads all 3 ANPHY-Sleep subjects with REAL PSG labels from .txt files
#   2. Trains three generative models on training split
#      NOTE: ALL three models now use WGAN-GP for fair comparison
#            (Classical GAN BCE was unfair — fixed here)
#   3. Uses each trained generator to augment minority sleep class (Wake)
#   4. Trains downstream RBF-SVM classifier on:
#      a) Real data only (baseline)
#      b) Real + Classical GAN augmented (WGAN-GP)
#      c) Real + QGAN Noiseless augmented (WGAN-GP)
#      d) Real + QGAN Noisy augmented (WGAN-GP)
#   5. Reports accuracy, macro-F1, per-class F1, improvement over baseline
#   6. Saves downstream_augmentation_results.json
#      and downstream_augmentation_table.txt (LaTeX ready for paper)
#
# Run: python -m qgan.evaluate_augmentation
#
# Expected time: ~3-5 hours on local CPU (3 subjects x 3 models x 3 features)
# Quick preview: set EPOCHS = 10 at the top
#
# DATASET: ANPHY-Sleep (EPCTL01, EPCTL02, EPCTL03)
#   Labels: W=0 (Wake), N1=1, N2=2, N3=3, R=4 (REM)
#   L (Lights) epochs are excluded — not a sleep stage
#
# WHY WGAN-GP FOR ALL THREE MODELS:
#   The original conference paper used BCE for Classical GAN and WGAN-GP
#   for QGANs. Reviewer 2 correctly identified this as an unfair comparison.
#   For this downstream augmentation experiment, all three models use
#   WGAN-GP with identical discriminator architecture, making the
#   comparison architecturally fair. The only difference is the generator:
#   classical MLP vs quantum circuit.

import copy
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pyedflib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from qgan.config      import LEARNING_RATE, GRAD_CLIP, EEG_CHANNEL, EPOCH_SECONDS
from qgan.models_noise import (
    ClassicalGenerator,
    GeneratorArchC,
    ClassicalDiscriminator,
    add_data_noise,
)

# ================================================================
#  CONFIG
# ================================================================
SUBJECT_PATHS = [
    ("data/EPCTL01.edf", "data/EPCTL01.txt"),
    ("data/EPCTL02.edf", "data/EPCTL02.txt"),
    ("data/EPCTL03.edf", "data/EPCTL03.txt"),
]

FEATURE_SWEEP = [2, 3, 4]
N_QUBITS      = 6
N_LAYERS      = 2
BATCH_SIZE    = 16
LAMBDA_GP     = 10      # WGAN-GP gradient penalty — same for ALL three models
NOISE_LEVEL   = 0.1
EPOCHS        = 50      # set to 10 for quick preview
AUG_RATIO     = 1.0     # generate minority_count * AUG_RATIO synthetic samples
RANDOM_STATE  = 42

LABEL_MAP   = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
LABEL_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

OUT_JSON  = "downstream_augmentation_results.json"
OUT_LATEX = "downstream_augmentation_table.txt"

LR_G = LEARNING_RATE
LR_D = LEARNING_RATE * 5.0


# ================================================================
#  LOAD ONE SUBJECT — EDF features + real PSG labels
#  Handles off-by-one mismatch gracefully
# ================================================================
def load_subject_with_labels(edf_path: str, txt_path: str,
                              n_features: int):
    with pyedflib.EdfReader(edf_path) as f:
        signal = f.readSignal(EEG_CHANNEL)
        sr     = f.getSampleFrequency(EEG_CHANNEL)

    samples_per_epoch = int(EPOCH_SECONDS * sr)
    n_epochs_edf      = len(signal) // samples_per_epoch
    signal            = signal[:n_epochs_edf * samples_per_epoch]
    epochs            = signal.reshape(n_epochs_edf, samples_per_epoch)

    all_feats = np.array(
        [[e.mean(), e.std(), e.min(), e.max()] for e in epochs],
        dtype=np.float32
    )
    feats = all_feats[:, :n_features]

    with open(txt_path) as f:
        rows = [line.strip().split() for line in f if line.strip()]

    # Handle off-by-one mismatch
    if len(rows) != n_epochs_edf:
        diff = len(rows) - n_epochs_edf
        if diff > 0:
            rows = rows[:n_epochs_edf]
        else:
            raise ValueError(
                f"Epoch count mismatch: EDF has {n_epochs_edf} epochs "
                f"but {txt_path} only has {len(rows)} rows."
            )

    keep_mask = []
    label_arr = []
    for row in rows:
        stage = row[0]
        if stage == "L":
            keep_mask.append(False)
            label_arr.append(-1)
        else:
            keep_mask.append(True)
            label_arr.append(LABEL_MAP.get(stage, -1))

    keep_mask   = np.array(keep_mask)
    label_arr   = np.array(label_arr)
    feats_kept  = feats[keep_mask]
    labels_kept = label_arr[keep_mask]

    # Normalise to [-1, 1]
    for col in range(n_features):
        lo, hi = feats_kept[:, col].min(), feats_kept[:, col].max()
        if hi - lo > 1e-8:
            feats_kept[:, col] = (
                2.0 * (feats_kept[:, col] - lo) / (hi - lo) - 1.0
            )
        else:
            feats_kept[:, col] = 0.0

    return feats_kept, labels_kept


# ================================================================
#  LOAD ALL SUBJECTS
# ================================================================
def load_all_subjects(n_features: int):
    all_features = []
    all_labels   = []

    print(f"\n  Loading {len(SUBJECT_PATHS)} subjects | {n_features} features")
    print(f"  {'─'*55}")

    for edf_path, txt_path in SUBJECT_PATHS:
        try:
            feats, labels = load_subject_with_labels(
                edf_path, txt_path, n_features)
            all_features.append(feats)
            all_labels.append(labels)
            name = os.path.basename(edf_path)
            print(f"    {name:<25} {len(feats)} epochs (L excluded)")
        except Exception as e:
            name = os.path.basename(edf_path)
            print(f"    {name:<25} SKIPPED — {e}")

    if not all_features:
        raise RuntimeError("No subjects loaded. Check your data files.")

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels,   axis=0)

    print(f"  {'─'*55}")
    print(f"  Total: {len(X)} epochs across {len(all_features)} subjects")
    print(f"  Class distribution:")
    for c in sorted(np.unique(y)):
        count = (y == c).sum()
        print(f"    {LABEL_NAMES[c]:<8} (class {c}): "
              f"{count:4d} samples ({100*count/len(y):.1f}%)")

    return X, y


# ================================================================
#  WGAN-GP GRADIENT PENALTY — shared by ALL three models
# ================================================================
def gradient_penalty(disc, real, fake):
    bs     = real.size(0)
    alpha  = torch.rand(bs, 1).float().expand(real.size())
    interp = (alpha * real.float() +
              (1 - alpha) * fake.float()).requires_grad_(True)
    d_out  = disc(interp)
    grads  = torch.autograd.grad(
        outputs=d_out, inputs=interp,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True, retain_graph=True
    )[0]
    return ((grads.view(bs, -1).norm(2, dim=1) - 1) ** 2).mean()


# ================================================================
#  WGAN-GP TRAINING LOOP — works for both Classical and Quantum generator
#
#  KEY CHANGE FROM CONFERENCE PAPER:
#  Classical GAN now uses WGAN-GP (same as QGANs) for fair comparison.
#  The only architectural difference is the generator type:
#    - Classical: MLP (ClassicalGenerator)
#    - Quantum:   PQC (GeneratorArchC)
#  Everything else — discriminator, loss, optimizer betas — is identical.
# ================================================================
def train_with_wgangp(generator, n_features, data_tensor,
                      n_epochs, label, is_quantum):
    print(f"    Training {label} with WGAN-GP ({n_epochs} epochs)...")

    disc  = ClassicalDiscriminator(input_dim=N_QUBITS)
    opt_g = torch.optim.Adam(generator.parameters(),
                              lr=LR_G, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(),
                              lr=LR_D, betas=(0.0, 0.9))

    loader = DataLoader(TensorDataset(data_tensor),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    best_d  = float("inf")
    best_gs = copy.deepcopy(generator.state_dict())

    for epoch in range(n_epochs):
        for (batch,) in loader:
            real = batch.float()
            bs   = real.shape[0]

            if real.shape[-1] < N_QUBITS:
                pad      = torch.zeros(bs, N_QUBITS - real.shape[-1])
                real_pad = torch.cat([real, pad], dim=-1)
            else:
                real_pad = real

            # Generator: 2 steps
            for _ in range(2):
                fake   = generator(torch.randn(bs, n_features))
                g_loss = -disc(fake).mean()
                opt_g.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    generator.parameters(), GRAD_CLIP)
                opt_g.step()

            # Discriminator: 1 step + gradient penalty
            fake   = generator(torch.randn(bs, n_features)).detach()
            gp     = gradient_penalty(disc, real_pad, fake)
            d_loss = (-disc(real_pad).mean() + disc(fake).mean()
                      + LAMBDA_GP * gp)
            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), GRAD_CLIP)
            opt_d.step()

        if d_loss.item() < best_d:
            best_d  = d_loss.item()
            best_gs = copy.deepcopy(generator.state_dict())

        if (epoch + 1) % 10 == 0 or (epoch + 1) == n_epochs:
            print(f"      Epoch [{epoch+1:3d}/{n_epochs}] "
                  f"D:{d_loss.item():.4f}")

    generator.load_state_dict(best_gs)
    return generator


# ================================================================
#  GENERATE SYNTHETIC SAMPLES
# ================================================================
def generate_synthetic(generator, n_generate, n_features, label):
    generator.eval()
    all_fake  = []
    generated = 0

    with torch.no_grad():
        while generated < n_generate:
            this_batch = min(50, n_generate - generated)
            z    = torch.randn(this_batch, n_features)
            fake = generator(z)
            fake = fake[:, :n_features].numpy()
            all_fake.append(fake)
            generated += this_batch

    generator.train()
    result = np.concatenate(all_fake, axis=0)[:n_generate]
    print(f"    Generated {len(result)} synthetic samples [{label}]")
    return result


# ================================================================
#  DOWNSTREAM SVM CLASSIFIER
# ================================================================
def evaluate_downstream(X_train, y_train, X_test, y_test, label):
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    clf = SVC(kernel="rbf", C=1.0, gamma="scale",
              random_state=RANDOM_STATE, class_weight="balanced")
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    acc    = round(accuracy_score(y_test, y_pred), 4)
    f1_mac = round(f1_score(y_test, y_pred, average="macro",
                            zero_division=0), 4)

    classes    = sorted(np.unique(np.concatenate([y_test, y_train])))
    f1_per_arr = f1_score(y_test, y_pred, labels=classes,
                          average=None, zero_division=0)
    f1_per     = {LABEL_NAMES[c]: round(float(v), 4)
                  for c, v in zip(classes, f1_per_arr)}

    print(f"    [{label:<28}] Acc={acc:.4f}  MacroF1={f1_mac:.4f}")
    for stage, val in f1_per.items():
        print(f"      {stage:<8} F1={val:.4f}")

    return {
        "label":        label,
        "accuracy":     acc,
        "macro_f1":     f1_mac,
        "f1_per_class": f1_per,
        "n_train":      int(len(X_train)),
        "n_test":       int(len(X_test)),
    }


# ================================================================
#  MAIN
# ================================================================
def main():
    print(f"\n  {'='*70}")
    print(f"  DOWNSTREAM CLASSIFIER EVALUATION")
    print(f"  Dataset   : ANPHY-Sleep (EPCTL01, EPCTL02, EPCTL03)")
    print(f"  Labels    : Real PSG labels — W, N1, N2, N3, R (L excluded)")
    print(f"  Loss      : WGAN-GP for ALL three models (fair comparison)")
    print(f"  Classifier: RBF-SVM with balanced class weights")
    print(f"  Epochs    : {EPOCHS} per model")
    print(f"  {'='*70}\n")

    all_results = {}

    for n_features in FEATURE_SWEEP:
        print(f"\n  {'='*70}")
        print(f"  FEATURE COUNT: {n_features}")
        print(f"  {'='*70}")

        X, y = load_all_subjects(n_features)

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2,
            random_state=RANDOM_STATE, stratify=y
        )
        print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}")

        train_tensor = torch.from_numpy(X_train.astype(np.float32))

        # Minority class for augmentation
        class_counts   = {c: int((y_train == c).sum())
                          for c in np.unique(y_train)}
        minority_class = min(class_counts, key=class_counts.get)
        minority_count = class_counts[minority_class]
        n_generate     = int(minority_count * AUG_RATIO)

        print(f"  Minority: {LABEL_NAMES[minority_class]} "
              f"({minority_count} train samples) "
              f"→ generate {n_generate} synthetic")

        feat_results = {}

        # ── Baseline ──────────────────────────────────────────────
        print(f"\n  [Baseline] Real data only")
        baseline = evaluate_downstream(
            X_train, y_train, X_test, y_test, "Real Only (Baseline)")
        feat_results["baseline"] = baseline

        # ── Classical GAN + WGAN-GP ───────────────────────────────
        print(f"\n  [1/3] Classical GAN (WGAN-GP)")
        c_gen = ClassicalGenerator(latent_dim=n_features, output_dim=N_QUBITS)
        c_gen = train_with_wgangp(c_gen, n_features, train_tensor,
                                   EPOCHS, "Classical GAN", is_quantum=False)
        c_fake   = generate_synthetic(c_gen, n_generate, n_features,
                                       "Classical GAN")
        X_aug_c  = np.concatenate([X_train, c_fake])
        y_aug_c  = np.concatenate([y_train, np.full(n_generate, minority_class)])
        c_result = evaluate_downstream(X_aug_c, y_aug_c, X_test, y_test,
                                        "Real + Classical GAN")
        c_result["acc_improvement"] = round(
            c_result["accuracy"] - baseline["accuracy"], 4)
        c_result["f1_improvement"]  = round(
            c_result["macro_f1"] - baseline["macro_f1"], 4)
        feat_results["classical_gan_aug"] = c_result

        # ── QGAN Noiseless + WGAN-GP ──────────────────────────────
        print(f"\n  [2/3] QGAN Noiseless (WGAN-GP)")
        qn_gen = GeneratorArchC(n_qubits=N_QUBITS, n_layers=N_LAYERS)
        qn_gen = train_with_wgangp(qn_gen, n_features, train_tensor,
                                    EPOCHS, "QGAN Noiseless", is_quantum=True)
        qn_fake   = generate_synthetic(qn_gen, n_generate, n_features,
                                        "QGAN Noiseless")
        X_aug_qn  = np.concatenate([X_train, qn_fake])
        y_aug_qn  = np.concatenate([y_train, np.full(n_generate, minority_class)])
        qn_result = evaluate_downstream(X_aug_qn, y_aug_qn, X_test, y_test,
                                         "Real + QGAN Noiseless")
        qn_result["acc_improvement"] = round(
            qn_result["accuracy"] - baseline["accuracy"], 4)
        qn_result["f1_improvement"]  = round(
            qn_result["macro_f1"] - baseline["macro_f1"], 4)
        feat_results["qgan_noiseless_aug"] = qn_result

        # ── QGAN Noisy + WGAN-GP ──────────────────────────────────
        print(f"\n  [3/3] QGAN Noisy (WGAN-GP, noise on input data)")
        noisy_tensor = add_data_noise(train_tensor, NOISE_LEVEL)
        qnoisy_gen   = GeneratorArchC(n_qubits=N_QUBITS, n_layers=N_LAYERS)
        qnoisy_gen   = train_with_wgangp(qnoisy_gen, n_features, noisy_tensor,
                                          EPOCHS, "QGAN Noisy", is_quantum=True)
        qnoisy_fake   = generate_synthetic(qnoisy_gen, n_generate, n_features,
                                            "QGAN Noisy")
        X_aug_qnoisy  = np.concatenate([X_train, qnoisy_fake])
        y_aug_qnoisy  = np.concatenate([y_train,
                                         np.full(n_generate, minority_class)])
        qnoisy_result = evaluate_downstream(X_aug_qnoisy, y_aug_qnoisy,
                                             X_test, y_test, "Real + QGAN Noisy")
        qnoisy_result["acc_improvement"] = round(
            qnoisy_result["accuracy"] - baseline["accuracy"], 4)
        qnoisy_result["f1_improvement"]  = round(
            qnoisy_result["macro_f1"] - baseline["macro_f1"], 4)
        feat_results["qgan_noisy_aug"] = qnoisy_result

        all_results[f"{n_features}_features"] = feat_results

        # Summary print
        print(f"\n  SUMMARY — {n_features} features "
              f"(3 subjects, real PSG labels, WGAN-GP all models):")
        print(f"  {'Condition':<32} {'Acc':<10} {'MacroF1':<12} "
              f"{'DeltaAcc':<12} {'DeltaF1'}")
        print(f"  {'─'*72}")
        for key, lbl in [
            ("baseline",           "Real Only (Baseline)   "),
            ("classical_gan_aug",  "Real + Classical GAN   "),
            ("qgan_noiseless_aug", "Real + QGAN Noiseless  "),
            ("qgan_noisy_aug",     "Real + QGAN Noisy      "),
        ]:
            r     = feat_results[key]
            d_acc = f"{r['acc_improvement']:+.4f}" \
                    if key != "baseline" else "---"
            d_f1  = f"{r['f1_improvement']:+.4f}" \
                    if key != "baseline" else "---"
            print(f"  {lbl:<32} {r['accuracy']:<10} "
                  f"{r['macro_f1']:<12} {d_acc:<12} {d_f1}")

        with open(OUT_JSON, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Saved: {OUT_JSON}")

    write_latex_table(all_results)

    print(f"\n  {'='*70}")
    print(f"  COMPLETE")
    print(f"    {OUT_JSON}")
    print(f"    {OUT_LATEX}")
    print(f"  {'='*70}\n")


# ================================================================
#  LATEX TABLE
# ================================================================
def write_latex_table(all_results):
    lines = [
        "% ============================================================",
        "% Downstream SVM Evaluation — ANPHY-Sleep (EPCTL01-03)",
        "% All three models trained with WGAN-GP for fair comparison",
        "% Real PSG labels: W=Wake, N1, N2, N3, R=REM",
        "% L (Lights) epochs excluded",
        "% Classifier: RBF-SVM, balanced class weights",
        "% ============================================================",
        "",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Downstream sleep stage classification on ANPHY-Sleep "
        "(EPCTL01--03, 3 subjects) using real PSG stage labels "
        "(W, N1, N2, N3, REM). All three generative models trained with "
        "WGAN-GP for architectural fairness. Each augments the minority "
        "class (Wake) with synthetic samples. "
        "Classifier: RBF-SVM with balanced class weights. "
        "Bold = best per feature group. "
        "$\\Delta$ = change over real-data-only baseline.}",
        "\\label{tab:downstream}",
        "\\begin{tabular}{|c|l|c|c|c|c|}",
        "\\hline",
        "\\textbf{Feat.} & \\textbf{Training Data} & "
        "\\textbf{Acc} & $\\boldsymbol{\\Delta}$\\textbf{Acc} & "
        "\\textbf{Macro F1} & $\\boldsymbol{\\Delta}$\\textbf{F1} \\\\",
        "\\hline",
    ]

    for n_f in FEATURE_SWEEP:
        key     = f"{n_f}_features"
        results = all_results.get(key, {})
        if not results:
            continue

        row_order = [
            ("baseline",           "Real Only (Baseline)"),
            ("classical_gan_aug",  "Real + Classical GAN (WGAN-GP)"),
            ("qgan_noiseless_aug", "Real + QGAN Noiseless"),
            ("qgan_noisy_aug",     "Real + QGAN Noisy"),
        ]

        valid_keys = [k for k, _ in row_order if k in results]
        best_acc   = max(results[k]["accuracy"] for k in valid_keys)
        best_f1    = max(results[k]["macro_f1"] for k in valid_keys)

        first = True
        for result_key, label in row_order:
            if result_key not in results:
                continue
            r       = results[result_key]
            acc_str = f"{r['accuracy']:.4f}"
            f1_str  = f"{r['macro_f1']:.4f}"
            if r["accuracy"] == best_acc:
                acc_str = f"\\textbf{{{acc_str}}}"
            if r["macro_f1"] == best_f1:
                f1_str = f"\\textbf{{{f1_str}}}"

            d_acc = "---" if result_key == "baseline" \
                    else f"{r['acc_improvement']:+.4f}"
            d_f1  = "---" if result_key == "baseline" \
                    else f"{r['f1_improvement']:+.4f}"

            feat_cell = str(n_f) if first else ""
            first     = False
            lines.append(
                f"  {feat_cell} & {label} & "
                f"{acc_str} & {d_acc} & "
                f"{f1_str} & {d_f1} \\\\"
            )
        lines.append("  \\hline")

    lines += ["\\end{tabular}", "\\end{table}"]

    with open(OUT_LATEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX table saved: {OUT_LATEX}")


if __name__ == "__main__":
    main()