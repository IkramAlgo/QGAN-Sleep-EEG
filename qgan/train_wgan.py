# qgan/train_wgan.py
# Architecture 3: Hybrid QGAN + Classical GAN with WGAN-GP loss
# ---------------------------------------------------------------
# WHY THIS FILE EXISTS:
#   BCE results are saved in results.json — DO NOT TOUCH
#   This file saves to results_wgan.json only
#   Lets us compare BCE vs WGAN-GP as a separate experiment
#
# WHY WGAN-GP (research basis):
#   Gulrajani et al. 2017 — original WGAN-GP paper, lambda=10 standard
#   Tian et al. 2025 EPJ Quantum Technology — WGAN-GP for hybrid QGAN
#   Zhang et al. 2021 IEEE — WGAN-GP outperforms BCE on EEG data
#   Proven best for biomedical GAN training, no mode collapse
#
# KEY DIFFERENCES FROM train.py (BCE):
#   1. Loss: Wasserstein distance + gradient penalty (no BCE, no sigmoid)
#   2. Adam betas: (0.0, 0.9) instead of (0.5, 0.999) — WGAN-GP standard
#   3. Discriminator trained BEFORE generator each step
#   4. Classification threshold: 0.0 (raw scores) not 0.5 (probabilities)
#   5. ClassicalDiscriminator: no Sigmoid at output (raw scores needed)
#   6. Saves to results_wgan.json — results.json untouched
#
# Run: python -m qgan.train_wgan

import torch
import torch.nn as nn
import copy
import time
import json
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

from qgan.config import (EPOCHS, BATCH_SIZE, LEARNING_RATE, LR_STEP_SIZE,
                          LR_GAMMA, GRAD_CLIP, EVAL_EVERY, EVAL_SAMPLES,
                          FEATURE_SWEEP, ALL_FEATURE_NAMES)
from qgan.models import GeneratorQuantumCircuit
from qgan.classical_baseline import ClassicalGenerator
from qgan.data_loader import get_data_loader

# ── WGAN-GP constant ────────────────────────────────────────────────────────
LAMBDA_GP = 10   # gradient penalty weight — standard from Gulrajani et al. 2017


# ── Discriminator WITHOUT Sigmoid ───────────────────────────────────────────
# WGAN-GP critic must output raw unbounded scores, not probabilities.
# We define a fresh class here so models.py / classical_baseline.py are untouched.

class WGANCritic(nn.Module):
    """Classical MLP critic for WGAN-GP. No Sigmoid at output."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            # NO Sigmoid — WGAN-GP requires raw scores
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x.float())


# ── Gradient penalty ────────────────────────────────────────────────────────

def gradient_penalty(critic, real, fake):
    """
    Enforces 1-Lipschitz constraint on the critic.
    Interpolates between real and fake, checks that gradient norm ≈ 1.
    lambda=10 penalty on deviation — standard from original paper.
    """
    bs = real.size(0)
    alpha = torch.rand(bs, 1).expand_as(real)
    interpolates = (alpha * real.float() +
                    (1.0 - alpha) * fake.float().detach()).requires_grad_(True)
    d_interp = critic(interpolates)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(bs, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ── WGAN-GP loss functions ───────────────────────────────────────────────────

def critic_loss(critic, real, fake):
    """
    Critic wants: high score for real, low score for fake.
    Wasserstein distance = E[real scores] - E[fake scores].
    We minimise the negative of that + gradient penalty.
    """
    gp = gradient_penalty(critic, real, fake)
    loss = (-critic(real.float()).mean()
            + critic(fake.detach().float()).mean()
            + LAMBDA_GP * gp)
    return loss


def generator_loss(critic, fake):
    """
    Generator wants critic to give high scores to its outputs.
    Minimise negative mean score = maximise mean score.
    """
    return -critic(fake.float()).mean()


# ── MAE metrics (identical to train.py) ─────────────────────────────────────

def mae_metrics(generator, loader, n_features):
    generator.eval()
    real_batches = []
    with torch.no_grad():
        for batch in loader:
            real_batches.append(batch[0])
            if len(real_batches) * BATCH_SIZE >= EVAL_SAMPLES:
                break
        real = torch.cat(real_batches)[:EVAL_SAMPLES]
        fake = generator(torch.randn(EVAL_SAMPLES, n_features))
    generator.train()
    return {
        "mean_MAE": round(torch.abs(real.mean(0) - fake.mean(0)).mean().item(), 4),
        "std_MAE":  round(torch.abs(real.std(0)  - fake.std(0) ).mean().item(), 4),
    }


# ── Classification metrics — threshold at 0.0 (raw Wasserstein scores) ──────

def classification_metrics(generator, critic, loader, n_features):
    """
    WGAN-GP scores are unbounded real numbers.
    Positive → critic thinks real.
    Negative → critic thinks fake.
    Threshold at 0.0, not 0.5.
    """
    generator.eval()
    critic.eval()
    real_batches = []
    with torch.no_grad():
        for batch in loader:
            real_batches.append(batch[0])
            if len(real_batches) * BATCH_SIZE >= EVAL_SAMPLES:
                break
        real = torch.cat(real_batches)[:EVAL_SAMPLES]
        fake = generator(torch.randn(len(real), n_features))
        rs = critic(real.float()).squeeze()
        fs = critic(fake.float()).squeeze()

    scores = torch.cat([rs, fs]).detach().numpy()
    labels = np.array([1] * len(real) + [0] * len(fake))
    preds  = (scores > 0.0).astype(int)   # ← 0.0 threshold for WGAN-GP
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    generator.train()
    critic.train()
    return {
        "Accuracy":    round(accuracy_score(labels, preds), 4),
        "Precision":   round(precision_score(labels, preds, zero_division=0), 4),
        "Sensitivity": round(recall_score(labels, preds, zero_division=0), 4),
        "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4),
        "F1":          round(f1_score(labels, preds, zero_division=0), 4),
    }


# ── Training loop ────────────────────────────────────────────────────────────

def train(generator, critic, loader, n_features, name):
    # WGAN-GP Adam settings: betas=(0.0, 0.9) as per original paper
    # discriminator 5x faster — same as BCE experiment for fair comparison
    opt_g = torch.optim.Adam(generator.parameters(),
                              lr=LEARNING_RATE,       betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(),
                              lr=LEARNING_RATE * 5.0, betas=(0.0, 0.9))
    sch_g = torch.optim.lr_scheduler.StepLR(opt_g, LR_STEP_SIZE, LR_GAMMA)
    sch_c = torch.optim.lr_scheduler.StepLR(opt_c, LR_STEP_SIZE, LR_GAMMA)

    history = {"gen_loss": [], "disc_loss": [], "mean_MAE": [],
               "std_MAE": [], "mae_epochs": [], "times": []}

    best_c_loss  = float("inf")
    best_c_state = copy.deepcopy(critic.state_dict())
    best_g_state = copy.deepcopy(generator.state_dict())

    print(f"\n  [{name}] — WGAN-GP loss (lambda={LAMBDA_GP})")

    for epoch in range(EPOCHS):
        t0 = time.time()
        g_losses, c_losses = [], []

        for batch in loader:
            real = batch[0]
            bs   = real.shape[0]

            # ── 1. Train critic first (standard WGAN-GP practice) ──────────
            fake    = generator(torch.randn(bs, n_features))
            c_loss  = critic_loss(critic, real, fake)
            opt_c.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
            opt_c.step()

            # ── 2. Train generator ─────────────────────────────────────────
            fake   = generator(torch.randn(bs, n_features))
            g_loss = generator_loss(critic, fake)
            opt_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
            opt_g.step()

            g_losses.append(g_loss.item())
            c_losses.append(c_loss.item())

        sch_g.step(); sch_c.step()

        avg_g   = np.mean(g_losses)
        avg_c   = np.mean(c_losses)
        elapsed = time.time() - t0

        history["gen_loss"].append(float(avg_g))
        history["disc_loss"].append(float(avg_c))
        history["times"].append(elapsed)

        if avg_c < best_c_loss:
            best_c_loss  = avg_c
            best_c_state = copy.deepcopy(critic.state_dict())
            best_g_state = copy.deepcopy(generator.state_dict())

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

    critic.load_state_dict(best_c_state)
    generator.load_state_dict(best_g_state)
    history["avg_time"] = round(float(np.mean(history["times"])), 4)
    return history, generator, critic


# ── Experiment runner ────────────────────────────────────────────────────────

def run_experiment(n_features):
    print(f"\n{'='*60}")
    print(f"  WGAN-GP EXPERIMENT: {n_features} features "
          f"({ALL_FEATURE_NAMES[:n_features]})")
    print(f"{'='*60}")

    loader = get_data_loader(n_features)

    # Hybrid QGAN — quantum generator + WGANCritic (no sigmoid)
    q_gen    = GeneratorQuantumCircuit(n_qubits=n_features)
    q_critic = WGANCritic(input_dim=n_features)
    q_hist, q_gen, q_critic = train(q_gen, q_critic, loader, n_features,
                                     "Hybrid QGAN + WGAN-GP")

    # Classical GAN — classical generator + WGANCritic (no sigmoid)
    c_gen    = ClassicalGenerator(n_features)
    c_critic = WGANCritic(input_dim=n_features)
    c_hist, c_gen, c_critic = train(c_gen, c_critic, loader, n_features,
                                     "Classical GAN + WGAN-GP")

    q_clf = classification_metrics(q_gen, q_critic, loader, n_features)
    c_clf = classification_metrics(c_gen, c_critic, loader, n_features)

    print(f"\n  Results ({n_features} features) — WGAN-GP:")
    print(f"  HybridQGAN — MeanMAE:{q_hist['mean_MAE'][-1]:.4f} "
          f"StdMAE:{q_hist['std_MAE'][-1]:.4f} "
          f"Acc:{q_clf['Accuracy']:.4f} F1:{q_clf['F1']:.4f} "
          f"Spec:{q_clf['Specificity']:.4f} AvgTime:{q_hist['avg_time']:.1f}s")
    print(f"  Classical  — MeanMAE:{c_hist['mean_MAE'][-1]:.4f} "
          f"StdMAE:{c_hist['std_MAE'][-1]:.4f} "
          f"Acc:{c_clf['Accuracy']:.4f} F1:{c_clf['F1']:.4f} "
          f"Spec:{c_clf['Specificity']:.4f} AvgTime:{c_hist['avg_time']:.1f}s")

    return {
        "n_features":    n_features,
        "feature_names": ALL_FEATURE_NAMES[:n_features],
        "loss":          "WGAN-GP",
        "lambda_gp":     LAMBDA_GP,
        "qgan": {
            "history": {k: [float(v) for v in vals] if isinstance(vals, list) else float(vals)
                        for k, vals in q_hist.items()},
            "clf": q_clf,
        },
        "classical": {
            "history": {k: [float(v) for v in vals] if isinstance(vals, list) else float(vals)
                        for k, vals in c_hist.items()},
            "clf": c_clf,
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Architecture 3: WGAN-GP")
    print("  BCE results (results.json) are NOT touched")
    print("  Output → results_wgan.json")
    print("="*60)

    all_results = []
    for n_features in FEATURE_SWEEP:
        result = run_experiment(n_features)
        all_results.append(result)

    with open("results_wgan.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n  Saved WGAN-GP results to results_wgan.json")
    print("  BCE results in results.json are unchanged")
    print("  Run: python -m qgan.plot_wgan  to generate WGAN-GP figures")


if __name__ == "__main__":
    main()