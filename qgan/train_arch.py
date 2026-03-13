# qgan/train_arch.py
# Architecture ablation study: Arch B, C, D × BCE and WGAN-GP loss
#
# WHAT THIS RUNS:
#   Arch B (RZ→RY+RX gate sequence)  × BCE      → results_arch_B_bce.json
#   Arch B (RZ→RY+RX gate sequence)  × WGAN-GP  → results_arch_B_wgan.json
#   Arch C (6 qubits with 2 ancilla) × BCE      → results_arch_C_bce.json
#   Arch C (6 qubits with 2 ancilla) × WGAN-GP  → results_arch_C_wgan.json
#   Arch D (all-to-all entanglement)  × BCE      → results_arch_D_bce.json
#   Arch D (all-to-all entanglement)  × WGAN-GP  → results_arch_D_wgan.json
#
# SAFETY: This script NEVER touches results.json, results_wgan.json,
#         results_ibm.json. All outputs go to separate files.
#
# HOW TO RUN:
#   python -m qgan.train_arch
#
# TIME ESTIMATE (from your machine's existing benchmark):
#   Arch B BCE:     ~67s/epoch × 50 epochs = ~56 min  (same speed as Arch A)
#   Arch B WGAN-GP: ~15s/epoch × 50 epochs = ~12 min
#   Arch C BCE:     ~280s/epoch × 50 epochs = ~4 hours (WARNING: slow)
#   Arch C WGAN-GP: ~60s/epoch × 50 epochs = ~50 min
#   Arch D BCE:     ~80s/epoch × 50 epochs = ~67 min  (~15% slower than A)
#   Arch D WGAN-GP: ~17s/epoch × 50 epochs = ~14 min
#   TOTAL ESTIMATED: ~7-8 hours
#
# TO RUN ONLY SPECIFIC ARCHS (faster):
#   Comment out archs in EXPERIMENTS list at bottom of this file.

import torch
import copy
import time
import json
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

from qgan.models_arch import GeneratorArchB, GeneratorArchC, GeneratorArchD
from qgan.classical_baseline import ClassicalGenerator, ClassicalDiscriminator
from qgan.data_loader import get_data_loader

# ─── Hyperparameters (same as Arch A for fair comparison) ─────────────────────
EPOCHS        = 50
BATCH_SIZE    = 32
LR            = 0.00005
LR_STEP       = 10
LR_GAMMA      = 0.95
GRAD_CLIP     = 1.0
EVAL_EVERY    = 10
EVAL_SAMPLES  = 200
N_FEATURES    = 4           # all arch experiments run at 4 features only
WGAN_LAMBDA   = 10          # gradient penalty weight
WGAN_N_CRITIC = 5           # critic steps per generator step
WGAN_LR_DISC  = 0.0001      # critic LR (5x higher, per WGAN-GP paper)

BCE = torch.nn.BCELoss()


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def bce_loss(fake, real, disc):
    """Standard BCE GAN loss."""
    pred_real         = torch.sigmoid(disc(real))
    pred_fake         = torch.sigmoid(disc(fake.detach()))
    pred_fake_for_gen = torch.sigmoid(disc(fake))
    g_loss = BCE(pred_fake_for_gen, torch.ones_like(pred_fake_for_gen))
    d_loss = (BCE(pred_real, torch.ones_like(pred_real)) +
              BCE(pred_fake, torch.zeros_like(pred_fake)))
    return g_loss, d_loss


def gradient_penalty(critic, real, fake):
    """WGAN-GP gradient penalty (Gulrajani et al. 2017)."""
    bs   = real.size(0)
    eps  = torch.rand(bs, 1).expand_as(real)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interp = critic(interp)
    if d_interp.dim() > 1:
        d_interp = d_interp.mean(dim=1, keepdim=True)
    grads = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True
    )[0]
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# =============================================================================
# EVALUATION
# =============================================================================

def mae_metrics(generator, loader, n_features):
    generator.eval()
    real_batches = []
    with torch.no_grad():
        for batch in loader:
            real_batches.append(batch[0])
            if len(real_batches) * BATCH_SIZE >= EVAL_SAMPLES:
                break
        real = torch.cat(real_batches)[:EVAL_SAMPLES]
        fake = generator(torch.randn(EVAL_SAMPLES, n_features)).float()
    generator.train()
    return {
        "mean_MAE": round(torch.abs(real.mean(0) - fake.mean(0)).mean().item(), 4),
        "std_MAE":  round(torch.abs(real.std(0)  - fake.std(0) ).mean().item(), 4),
    }


def clf_metrics(generator, disc, loader, n_features, wgan=False):
    generator.eval()
    disc.eval()
    real_batches = []
    with torch.no_grad():
        for batch in loader:
            real_batches.append(batch[0])
            if len(real_batches) * BATCH_SIZE >= EVAL_SAMPLES:
                break
        real = torch.cat(real_batches)[:EVAL_SAMPLES]
        fake = generator(torch.randn(len(real), n_features)).float()
        rs = disc(real)
        fs = disc(fake)
        if not wgan:
            rs = torch.sigmoid(rs)
            fs = torch.sigmoid(fs)
        rs = rs.mean(1) if rs.dim() > 1 else rs
        fs = fs.mean(1) if fs.dim() > 1 else fs

    threshold = 0.0 if wgan else 0.5
    scores = torch.cat([rs, fs]).numpy()
    labels = np.array([1] * len(real) + [0] * len(fake))
    preds  = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    generator.train()
    disc.train()
    return {
        "Accuracy":    round(accuracy_score(labels, preds), 4),
        "Precision":   round(precision_score(labels, preds, zero_division=0), 4),
        "Sensitivity": round(recall_score(labels, preds, zero_division=0), 4),
        "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4),
        "F1":          round(f1_score(labels, preds, zero_division=0), 4),
    }


# =============================================================================
# TRAINING — BCE
# =============================================================================

def train_bce(generator, disc, loader, n_features, name):
    opt_g = torch.optim.Adam(list(generator.parameters()), lr=LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(list(disc.parameters()),      lr=LR, betas=(0.5, 0.999))
    sch_g = torch.optim.lr_scheduler.StepLR(opt_g, LR_STEP, LR_GAMMA)
    sch_d = torch.optim.lr_scheduler.StepLR(opt_d, LR_STEP, LR_GAMMA)

    history = {"gen_loss": [], "disc_loss": [], "mean_MAE": [],
               "std_MAE": [], "mae_epochs": [], "times": []}

    best_d_loss  = float("inf")
    best_g_state = copy.deepcopy(dict(zip(
        [n for n, _ in generator.named_parameters()],
        [p.data.clone() for p in generator.parameters()]
    )))
    best_d_state = copy.deepcopy(disc.state_dict())

    print(f"\n  [BCE | {name}]")

    for epoch in range(EPOCHS):
        t0 = time.time()
        g_losses, d_losses = [], []

        for batch in loader:
            real = batch[0]
            bs   = real.shape[0]

            # generator step
            fake      = generator(torch.randn(bs, n_features)).float()
            pred_fake_gen = torch.sigmoid(disc(fake))
            g_loss    = BCE(pred_fake_gen, torch.ones_like(pred_fake_gen))
            opt_g.zero_grad(); g_loss.backward()
            for p in generator.parameters():
                torch.nn.utils.clip_grad_norm_([p], GRAD_CLIP)
            opt_g.step()

            # discriminator step
            fake       = generator(torch.randn(bs, n_features)).detach().float()
            pred_real  = torch.sigmoid(disc(real))
            pred_fake  = torch.sigmoid(disc(fake))
            d_loss     = (BCE(pred_real, torch.ones_like(pred_real)) +
                          BCE(pred_fake, torch.zeros_like(pred_fake)))
            opt_d.zero_grad(); d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), GRAD_CLIP)
            opt_d.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        sch_g.step(); sch_d.step()

        avg_g = np.mean(g_losses)
        avg_d = np.mean(d_losses)
        elapsed = time.time() - t0

        history["gen_loss"].append(avg_g)
        history["disc_loss"].append(avg_d)
        history["times"].append(elapsed)

        if avg_d < best_d_loss:
            best_d_loss  = avg_d
            best_d_state = copy.deepcopy(disc.state_dict())

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            mae = mae_metrics(generator, loader, n_features)
            history["mean_MAE"].append(mae["mean_MAE"])
            history["std_MAE"].append(mae["std_MAE"])
            history["mae_epochs"].append(epoch + 1)
            print(f"    Epoch [{epoch+1:3d}/{EPOCHS}] G:{avg_g:.4f} D:{avg_d:.4f} "
                  f"MeanMAE:{mae['mean_MAE']:.4f} StdMAE:{mae['std_MAE']:.4f} "
                  f"Time:{elapsed:.1f}s")
        else:
            print(f"    Epoch [{epoch+1:3d}/{EPOCHS}] G:{avg_g:.4f} D:{avg_d:.4f} "
                  f"Time:{elapsed:.1f}s")

    disc.load_state_dict(best_d_state)
    history["avg_time"] = round(float(np.mean(history["times"])), 4)
    return history, generator, disc


# =============================================================================
# TRAINING — WGAN-GP
# =============================================================================

def train_wgan(generator, disc, loader, n_features, name):
    """
    WGAN-GP training. Critic trained N_CRITIC times per generator step.
    Adam betas (0.0, 0.9) per WGAN-GP paper.
    """
    opt_g = torch.optim.Adam(list(generator.parameters()),
                             lr=LR, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(list(disc.parameters()),
                             lr=WGAN_LR_DISC, betas=(0.0, 0.9))

    history = {"gen_loss": [], "critic_loss": [], "mean_MAE": [],
               "std_MAE": [], "mae_epochs": [], "times": []}

    print(f"\n  [WGAN-GP | {name}]")

    for epoch in range(EPOCHS):
        t0 = time.time()
        g_losses, c_losses = [], []

        for batch in loader:
            real = batch[0]
            bs   = real.shape[0]

            # ── CRITIC STEPS (N_CRITIC per generator step) ──────────────────
            for _ in range(WGAN_N_CRITIC):
                fake   = generator(torch.randn(bs, n_features)).detach().float()
                c_real = disc(real)
                c_fake = disc(fake)
                if c_real.dim() > 1: c_real = c_real.mean(dim=1)
                if c_fake.dim() > 1: c_fake = c_fake.mean(dim=1)
                gp     = gradient_penalty(disc, real, fake)
                c_loss = -(c_real.mean() - c_fake.mean()) + WGAN_LAMBDA * gp
                opt_c.zero_grad(); c_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), GRAD_CLIP)
                opt_c.step()
                c_losses.append(c_loss.item())

            # ── GENERATOR STEP ───────────────────────────────────────────────
            fake   = generator(torch.randn(bs, n_features)).float()
            c_fake = disc(fake)
            if c_fake.dim() > 1: c_fake = c_fake.mean(dim=1)
            g_loss = -c_fake.mean()
            opt_g.zero_grad(); g_loss.backward()
            for p in generator.parameters():
                torch.nn.utils.clip_grad_norm_([p], GRAD_CLIP)
            opt_g.step()
            g_losses.append(g_loss.item())

        elapsed = time.time() - t0
        avg_g = np.mean(g_losses)
        avg_c = np.mean(c_losses)

        history["gen_loss"].append(avg_g)
        history["critic_loss"].append(avg_c)
        history["times"].append(elapsed)

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            mae = mae_metrics(generator, loader, n_features)
            history["mean_MAE"].append(mae["mean_MAE"])
            history["std_MAE"].append(mae["std_MAE"])
            history["mae_epochs"].append(epoch + 1)
            print(f"    Epoch [{epoch+1:3d}/{EPOCHS}] G:{avg_g:.4f} C:{avg_c:.4f} "
                  f"MeanMAE:{mae['mean_MAE']:.4f} StdMAE:{mae['std_MAE']:.4f} "
                  f"Time:{elapsed:.1f}s")
        else:
            print(f"    Epoch [{epoch+1:3d}/{EPOCHS}] G:{avg_g:.4f} C:{avg_c:.4f} "
                  f"Time:{elapsed:.1f}s")

    history["avg_time"] = round(float(np.mean(history["times"])), 4)
    return history, generator, disc


# =============================================================================
# RUN ONE ARCHITECTURE EXPERIMENT
# =============================================================================

def run_experiment(arch_name, GeneratorClass, arch_kwargs, loss_type):
    """
    Train one architecture with one loss function.
    Saves to results_arch_{arch_name}_{loss_type}.json
    Never overwrites any existing results files.
    """
    out_file = f"results_arch_{arch_name}_{loss_type}.json"
    print(f"\n{'='*65}")
    print(f"  ARCH {arch_name} | {loss_type.upper()} | {N_FEATURES} features")
    print(f"  Output: {out_file}")
    print(f"{'='*65}")

    loader = get_data_loader(N_FEATURES)

    # ── Instantiate models ───────────────────────────────────────────────────
    generator = GeneratorClass(**arch_kwargs)

    if loss_type == "wgan":
        # WGAN-GP uses raw score critic — no sigmoid — classical discriminator
        # Remove final sigmoid by using ClassicalDiscriminator directly
        # (ClassicalDiscriminator has no sigmoid at output — correct for WGAN-GP)
        disc = ClassicalDiscriminator(N_FEATURES)
    else:
        disc = ClassicalDiscriminator(N_FEATURES)

    # ── Train ────────────────────────────────────────────────────────────────
    if loss_type == "bce":
        history, generator, disc = train_bce(
            generator, disc, loader, N_FEATURES, arch_name)
    else:
        history, generator, disc = train_wgan(
            generator, disc, loader, N_FEATURES, arch_name)

    # ── Classification metrics ───────────────────────────────────────────────
    print(f"\n  Computing classification metrics...")
    q_clf = clf_metrics(generator, disc, loader, N_FEATURES,
                        wgan=(loss_type == "wgan"))

    print(f"\n  {arch_name} {loss_type.upper()} Results:")
    print(f"    MeanMAE:  {history['mean_MAE'][-1]:.4f}")
    print(f"    StdMAE:   {history['std_MAE'][-1]:.4f}")
    print(f"    Accuracy: {q_clf['Accuracy']:.4f}")
    print(f"    Spec:     {q_clf['Specificity']:.4f}")
    print(f"    F1:       {q_clf['F1']:.4f}")
    print(f"    AvgTime:  {history['avg_time']:.1f}s/epoch")

    result = {
        "arch":      arch_name,
        "loss_type": loss_type,
        "n_features": N_FEATURES,
        "history": {
            k: [float(v) for v in vals] if isinstance(vals, list) else float(vals)
            for k, vals in history.items()
        },
        "clf": q_clf,
    }

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_file}")

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── Define experiments ───────────────────────────────────────────────────
    # Each tuple: (arch_name, GeneratorClass, constructor_kwargs, loss_type)
    # Comment out any you don't want to run to save time.

    EXPERIMENTS = [
        # ARCH D first — lowest risk, fastest comparison to Arch A
        ("D", GeneratorArchD, {"n_qubits": N_FEATURES}, "bce"),
        ("D", GeneratorArchD, {"n_qubits": N_FEATURES}, "wgan"),

        # ARCH B — different gate sequence
        ("B", GeneratorArchB, {"n_qubits": N_FEATURES}, "bce"),
        ("B", GeneratorArchB, {"n_qubits": N_FEATURES}, "wgan"),

        # ARCH C — 6 qubits (SLOWEST — run last)
        # WARNING: BCE will take ~4 hours. Run separately if needed.
        ("C", GeneratorArchC, {"n_features": N_FEATURES}, "wgan"),  # WGAN-GP first (faster)
        #("C", GeneratorArchC, {"n_features": N_FEATURES}, "bce"),
    ]

    all_results = []
    total_start = time.time()

    for arch_name, GenClass, kwargs, loss_type in EXPERIMENTS:
        result = run_experiment(arch_name, GenClass, kwargs, loss_type)
        all_results.append(result)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  ARCHITECTURE ABLATION SUMMARY — 4 features")
    print(f"{'='*75}")
    print(f"  {'Arch':<6} {'Loss':<8} {'MeanMAE':<10} {'StdMAE':<10} "
          f"{'Acc':<8} {'Spec':<8} {'F1':<8} {'Time/ep':<10}")
    print(f"  {'-'*71}")

    # Also print existing Arch A results for reference
    print(f"  {'A':<6} {'BCE':<8} {'0.5247':<10} {'0.0797':<10} "
          f"{'0.5000':<8} {'0.0000':<8} {'0.6667':<8} {'67.6s':<10}  [existing]")
    print(f"  {'A':<6} {'WGAN-GP':<8} {'0.4863':<10} {'0.0808':<10} "
          f"{'0.8250':<8} {'0.7300':<8} {'0.8402':<8} {'14.6s':<10}  [existing]")
    print(f"  {'-'*71}")

    for r in all_results:
        h   = r["history"]
        clf = r["clf"]
        print(f"  {r['arch']:<6} {r['loss_type'].upper():<8} "
              f"{h['mean_MAE'][-1]:<10.4f} {h['std_MAE'][-1]:<10.4f} "
              f"{clf['Accuracy']:<8.4f} {clf['Specificity']:<8.4f} "
              f"{clf['F1']:<8.4f} {h['avg_time']:.1f}s")

    elapsed_total = time.time() - total_start
    print(f"\n  Total training time: {elapsed_total/3600:.1f} hours")
    print(f"\n  Files saved:")
    for r in all_results:
        print(f"    results_arch_{r['arch']}_{r['loss_type']}.json")
    print(f"\n  Next step: python -m qgan.plot_arch  (to be created)")


if __name__ == "__main__":
    main()