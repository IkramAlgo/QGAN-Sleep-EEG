# qgan/train_ibm.py
# QGAN Arch C on IBM QPU
# LOCAL TEST : python -m qgan.train_ibm           (3 epochs, noise sim)
# ARC RUN    : python -m qgan.train_ibm --arc      (50 epochs, real QPU)
#
# Prerequisites:
#   1. Run ibm_quantum_setup.py once to save credentials
#   2. ibm_credentials.txt must exist in project root

import os
import sys
import copy
import time
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

from qgan.config      import LEARNING_RATE, GRAD_CLIP
from qgan.data_loader import load_sleep_edf
from qgan.models_ibm  import GeneratorArchC, ClassicalDiscriminator

# ================================================================
#  CONFIG
# ================================================================
N_FEATURES  = 4       # EEG features used (Mean, Std, Min, Max)
N_QUBITS    = 6       # Arch C: 6 qubits
N_LAYERS    = 2
SHOTS       = 1024
BATCH_SIZE  = 8       # small — each sample = one QPU circuit call
LAMBDA_GP   = 10      # WGAN-GP gradient penalty weight
LOCAL_EPOCHS = 3
ARC_EPOCHS   = 50
# ================================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc", action="store_true",
                        help="Run on real QPU with 50 epochs (ARC mode)")
    parser.add_argument("--shots", type=int, default=SHOTS)
    return parser.parse_args()


# ================================================================
#  WGAN-GP GRADIENT PENALTY
# ================================================================
def gradient_penalty(disc, real, fake):
    bs    = real.size(0)
    alpha = torch.rand(bs, 1).float().expand(real.size())
    interp = (alpha * real.float() + (1 - alpha) * fake.float()).requires_grad_(True)
    d_out  = disc(interp)
    grads  = torch.autograd.grad(
        outputs=d_out, inputs=interp,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True, retain_graph=True
    )[0]
    grads = grads.view(bs, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


# ================================================================
#  METRICS
# ================================================================
def compute_mae(generator, data):
    generator.eval()
    with torch.no_grad():
        real     = data[:100].float()
        fake_all = generator(torch.randn(100, N_FEATURES))
        fake     = fake_all[:, :N_FEATURES]    # first 4 qubits = EEG features
    generator.train()
    return {
        "mean_MAE": round(torch.abs(real.mean(0) - fake.mean(0)).mean().item(), 4),
        "std_MAE":  round(torch.abs(real.std(0)  - fake.std(0) ).mean().item(), 4),
    }


def compute_clf(generator, disc, data):
    generator.eval(); disc.eval()
    n = min(100, len(data))
    with torch.no_grad():
        real = data[:n].float()
        fake = generator(torch.randn(n, N_FEATURES))

        # Pad real from 4 to 6 dims for discriminator
        if real.shape[-1] < N_QUBITS:
            pad     = torch.zeros(n, N_QUBITS - real.shape[-1])
            real_in = torch.cat([real, pad], dim=-1)
        else:
            real_in = real

        rs = disc(real_in).squeeze()
        fs = disc(fake).squeeze()

    scores = torch.cat([rs, fs]).detach().numpy()
    labels = np.array([1]*n + [0]*n)
    preds  = (scores > 0.0).astype(int)     # WGAN-GP threshold = 0, not 0.5

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    generator.train(); disc.train()
    return {
        "Accuracy":    round(accuracy_score(labels, preds), 4),
        "Precision":   round(precision_score(labels, preds, zero_division=0), 4),
        "Sensitivity": round(recall_score(labels, preds, zero_division=0), 4),
        "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4),
        "F1":          round(f1_score(labels, preds, zero_division=0), 4),
    }


# ================================================================
#  TRAINING LOOP
# ================================================================
def train(generator, disc, loader, data, n_epochs):
    # WGAN-GP uses (0.0, 0.9) betas — not (0.5, 0.999)
    opt_g = torch.optim.Adam(generator.parameters(),
                              lr=LEARNING_RATE,       betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(),
                              lr=LEARNING_RATE * 5.0, betas=(0.0, 0.9))

    history = {
        "gen_loss":   [], "disc_loss":  [],
        "mean_MAE":   [], "std_MAE":    [],
        "mae_epochs": [], "times":      []
    }
    best_d  = float("inf")
    best_gs = copy.deepcopy(generator.state_dict())
    best_ds = copy.deepcopy(disc.state_dict())

    print(f"\n  {'='*60}")
    print(f"  QGAN Arch C | IBM QPU Training")
    print(f"  Generator : {N_QUBITS} qubits | ring CNOT | RX->CNOT->RY")
    print(f"  Device    : {generator.device_label}")
    print(f"  Epochs    : {n_epochs} | Shots: {SHOTS} | Batch: {BATCH_SIZE}")
    print(f"  Loss      : WGAN-GP (lambda={LAMBDA_GP})")
    print(f"  {'='*60}\n")

    for epoch in range(n_epochs):
        t0 = time.time()
        g_losses, d_losses = [], []

        for batch in loader:
            real = batch[0].float()
            bs   = real.shape[0]

            # Pad real data from 4 to 6 dims for discriminator
            if real.shape[-1] < N_QUBITS:
                pad         = torch.zeros(bs, N_QUBITS - real.shape[-1])
                real_padded = torch.cat([real, pad], dim=-1)
            else:
                real_padded = real

            # Generator — 3 steps per 1 discriminator step
            for _ in range(3):
                fake   = generator(torch.randn(bs, N_FEATURES))
                g_loss = -disc(fake).mean()
                opt_g.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
                opt_g.step()
                g_losses.append(g_loss.item())

            # Discriminator — 1 step
            fake   = generator(torch.randn(bs, N_FEATURES)).detach()
            gp     = gradient_penalty(disc, real_padded, fake)
            d_loss = (-disc(real_padded).mean()
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

        if avg_d < best_d:
            best_d  = avg_d
            best_gs = copy.deepcopy(generator.state_dict())
            best_ds = copy.deepcopy(disc.state_dict())

        mae = compute_mae(generator, data)
        history["mean_MAE"].append(mae["mean_MAE"])
        history["std_MAE"].append(mae["std_MAE"])
        history["mae_epochs"].append(epoch + 1)

        print(f"  Epoch [{epoch+1:3d}/{n_epochs}] "
              f"G:{avg_g:+.4f} C:{avg_d:+.4f} "
              f"MeanMAE:{mae['mean_MAE']:.4f} "
              f"StdMAE:{mae['std_MAE']:.4f} "
              f"Time:{elapsed:.1f}s")

        # Checkpoint every 10 epochs — safety net for long ARC runs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == n_epochs:
            _checkpoint(history, epoch + 1)

    generator.load_state_dict(best_gs)
    disc.load_state_dict(best_ds)
    history["avg_time_per_epoch"] = round(float(np.mean(history["times"])), 2)
    return history, generator, disc


def _checkpoint(history, epoch):
    path = f"checkpoint_ibm_epoch{epoch}.json"
    with open(path, "w") as f:
        json.dump({"epoch": epoch, "history": history}, f, indent=2)
    print(f"  [Checkpoint saved: {path}]")


# ================================================================
#  MAIN
# ================================================================
def main():
    args = parse_args()

    if args.arc:
        n_epochs     = ARC_EPOCHS
        use_real_qpu = True
        out_file     = "results_ibm_qpu.json"
        print(f"\n  MODE: ARC — {n_epochs} epochs | Real IBM QPU")
    else:
        n_epochs     = LOCAL_EPOCHS
        use_real_qpu = False
        out_file     = "results_ibm_noise_sim.json"
        print(f"\n  MODE: LOCAL TEST — {n_epochs} epochs | Noise simulator")

    # Load data
    data   = load_sleep_edf()[:, :N_FEATURES]
    loader = DataLoader(TensorDataset(data),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"  Data: {len(data)} samples | {N_FEATURES} features")

    # Build models
    print("\n  Building models...")
    generator = GeneratorArchC(n_qubits=N_QUBITS, n_layers=N_LAYERS,
                               shots=args.shots, use_real_qpu=use_real_qpu)
    disc      = ClassicalDiscriminator(input_dim=N_QUBITS)

    # Train
    history, generator, disc = train(generator, disc, loader, data, n_epochs)

    # Final metrics
    print("\n  Computing final metrics...")
    clf = compute_clf(generator, disc, data)
    mae = compute_mae(generator, data)

    print(f"\n  FINAL RESULTS")
    print(f"  {'─'*45}")
    print(f"  Architecture : Arch C — 6-qubit ring CNOT WGAN-GP")
    print(f"  Device       : {generator.device_label}")
    print(f"  MeanMAE      : {mae['mean_MAE']}")
    print(f"  StdMAE       : {mae['std_MAE']}")
    for k, v in clf.items():
        print(f"  {k:<13}: {v}")
    print(f"  Avg Time     : {history['avg_time_per_epoch']}s/epoch")
    print(f"  {'─'*45}")

    results = {
        "architecture": "Arch C — 6-qubit ring CNOT RX->CNOT->RY WGAN-GP",
        "n_features":   N_FEATURES,
        "n_qubits":     N_QUBITS,
        "n_layers":     N_LAYERS,
        "shots":        args.shots,
        "epochs":       n_epochs,
        "device":       generator.device_label,
        "history":      history,
        "mae":          mae,
        "clf":          clf,
    }

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_file}")


if __name__ == "__main__":
    main()