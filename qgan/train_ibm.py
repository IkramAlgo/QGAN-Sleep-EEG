# qgan/train_ibm.py
# QGAN Arch C on IBM QPU — Feature Sweep [2, 3, 4]
# LOCAL TEST : python -m qgan.train_ibm           (3 epochs, noise sim)
# ARC RUN    : python -m qgan.train_ibm --arc      (50 epochs, real QPU)
#
# Runs experiment for each feature count: 2, 3, 4
# Saves one JSON per feature count + one combined results file
#
# Prerequisites:
#   1. Run ibm_quantum_setup.py once to save credentials
#   2. ibm_credentials.txt must exist in project root

import os
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

#  CONFIG
FEATURE_SWEEP = [2]#[2, 3, 4]   # run experiments for each feature count
FEATURE_NAMES = {
    2: ["Mean", "Std Dev"],
    #3: ["Mean", "Std Dev", "Min"],
    #4: ["Mean", "Std Dev", "Min", "Max"],
}
N_QUBITS     = 6       # Arch C: always 6 qubits regardless of features
N_LAYERS     = 2
SHOTS        = 64#512
BATCH_SIZE   = 8
LAMBDA_GP    = 10
LOCAL_EPOCHS = 3
ARC_EPOCHS   = 3#10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc", action="store_true",
                        help="Run on real QPU with 50 epochs (ARC mode)")
    parser.add_argument("--shots", type=int, default=SHOTS)
    return parser.parse_args()


#  WGAN-GP GRADIENT PENALTY
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


#  METRICS
def compute_mae(generator, data, n_features):
    generator.eval()
    with torch.no_grad():
        real     = data[:100, :n_features].float()
        fake_all = generator(torch.randn(100, n_features))
        fake     = fake_all[:, :n_features]   # first n_features qubits
    generator.train()
    return {
        "mean_MAE": round(torch.abs(real.mean(0) - fake.mean(0)).mean().item(), 4),
        "std_MAE":  round(torch.abs(real.std(0)  - fake.std(0) ).mean().item(), 4),
    }


def compute_clf(generator, disc, data, n_features):
    generator.eval(); disc.eval()
    n = min(100, len(data))
    with torch.no_grad():
        real = data[:n, :n_features].float()
        fake = generator(torch.randn(n, n_features))

        # Pad real from n_features to N_QUBITS for discriminator
        if real.shape[-1] < N_QUBITS:
            pad     = torch.zeros(n, N_QUBITS - real.shape[-1])
            real_in = torch.cat([real, pad], dim=-1)
        else:
            real_in = real

        rs = disc(real_in).squeeze()
        fs = disc(fake).squeeze()

    scores = torch.cat([rs, fs]).detach().numpy()
    labels = np.array([1]*n + [0]*n)
    preds  = (scores > 0.0).astype(int)    # WGAN-GP threshold = 0

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    generator.train(); disc.train()
    return {
        "Accuracy":    round(accuracy_score(labels, preds), 4),
        "Precision":   round(precision_score(labels, preds, zero_division=0), 4),
        "Sensitivity": round(recall_score(labels, preds, zero_division=0), 4),
        "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4),
        "F1":          round(f1_score(labels, preds, zero_division=0), 4),
    }


#  TRAINING LOOP
def train(generator, disc, loader, data, n_features, n_epochs):
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

    for epoch in range(n_epochs):
        t0 = time.time()
        g_losses, d_losses = [], []

        for batch in loader:
            real = batch[0].float()
            bs   = real.shape[0]

            # Pad real from n_features to N_QUBITS for discriminator
            if real.shape[-1] < N_QUBITS:
                pad         = torch.zeros(bs, N_QUBITS - real.shape[-1])
                real_padded = torch.cat([real, pad], dim=-1)
            else:
                real_padded = real

            # Generator — 3 steps per 1 discriminator step
            for _ in range(3):
                fake   = generator(torch.randn(bs, n_features))
                g_loss = -disc(fake).mean()
                opt_g.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
                opt_g.step()
                g_losses.append(g_loss.item())

            # Discriminator — 1 step
            fake   = generator(torch.randn(bs, n_features)).detach()
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

        mae = compute_mae(generator, data, n_features)
        history["mean_MAE"].append(mae["mean_MAE"])
        history["std_MAE"].append(mae["std_MAE"])
        history["mae_epochs"].append(epoch + 1)

        print(f"    Epoch [{epoch+1:3d}/{n_epochs}] "
              f"G:{avg_g:+.4f} C:{avg_d:+.4f} "
              f"MeanMAE:{mae['mean_MAE']:.4f} "
              f"StdMAE:{mae['std_MAE']:.4f} "
              f"Time:{elapsed:.1f}s")

        # Checkpoint every 10 epochs — safety net for ARC
        if (epoch + 1) % 10 == 0 or (epoch + 1) == n_epochs:
            _checkpoint(history, n_features, epoch + 1)

    generator.load_state_dict(best_gs)
    disc.load_state_dict(best_ds)
    history["avg_time_per_epoch"] = round(float(np.mean(history["times"])), 2)
    return history, generator, disc


def _checkpoint(history, n_features, epoch):
    path = f"checkpoint_ibm_feat{n_features}_epoch{epoch}.json"
    with open(path, "w") as f:
        json.dump({"n_features": n_features,
                   "epoch": epoch, "history": history}, f, indent=2)
    print(f"    [Checkpoint saved: {path}]")


#  MAIN
def main():
    args = parse_args()

    if args.arc:
        n_epochs     = ARC_EPOCHS
        use_real_qpu = True
        out_file     = "results_ibm_qpu.json"
        mode_label   = "ARC — Real IBM QPU"
    else:
        n_epochs     = LOCAL_EPOCHS
        use_real_qpu = False
        out_file     = "results_ibm_local.json"
        mode_label   = "LOCAL TEST — Noise Simulator"

    print(f"\n  {'='*62}")
    print(f"  QGAN Arch C | IBM QPU | Feature Sweep {FEATURE_SWEEP}")
    print(f"  Mode    : {mode_label}")
    print(f"  Epochs  : {n_epochs} per feature count")
    print(f"  Shots   : {args.shots}")
    print(f"  {'='*62}")

    # Load full data once — slice features per experiment
    all_data = load_sleep_edf()

    all_results = {}

    for n_features in FEATURE_SWEEP:
        feat_names = FEATURE_NAMES[n_features]
        print(f"\n  {'='*62}")
        print(f"  EXPERIMENT: {n_features} features {feat_names}")
        print(f"  {'='*62}")

        # Slice data to n_features
        data   = all_data[:, :n_features]
        loader = DataLoader(TensorDataset(data),
                            batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        print(f"  Data: {len(data)} samples | {n_features} features")

        # Build fresh models for each feature count
        print("  Building models...")
        generator = GeneratorArchC(n_qubits=N_QUBITS, n_layers=N_LAYERS,
                                   shots=args.shots, use_real_qpu=use_real_qpu)
        disc      = ClassicalDiscriminator(input_dim=N_QUBITS)

        print(f"  Device: {generator.device_label}")
        print(f"  Architecture: {N_QUBITS} qubits | ring CNOT | RX->CNOT->RY | WGAN-GP\n")

        # Train
        history, generator, disc = train(
            generator, disc, loader, data, n_features, n_epochs
        )

        # Final metrics
        clf = compute_clf(generator, disc, data, n_features)
        mae = compute_mae(generator, data, n_features)

        print(f"\n  Results — {n_features} features:")
        print(f"  MeanMAE:{mae['mean_MAE']}  StdMAE:{mae['std_MAE']}  "
              f"Acc:{clf['Accuracy']}  Spec:{clf['Specificity']}  F1:{clf['F1']}")

        all_results[f"{n_features}_features"] = {
            "n_features":    n_features,
            "feature_names": feat_names,
            "architecture":  "Arch C — 6-qubit ring CNOT RX->CNOT->RY WGAN-GP",
            "n_qubits":      N_QUBITS,
            "n_layers":      N_LAYERS,
            "shots":         args.shots,
            "epochs":        n_epochs,
            "device":        generator.device_label,
            "history":       history,
            "mae":           mae,
            "clf":           clf,
        }

        # Save per-feature result immediately — do not lose it
        per_file = out_file.replace(".json", f"_{n_features}f.json")
        with open(per_file, "w") as f:
            json.dump(all_results[f"{n_features}_features"], f, indent=2)
        print(f"  Saved: {per_file}")

    # Save combined results
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Combined results saved: {out_file}")

    # Print summary table
    print(f"\n  {'='*70}")
    print(f"  SUMMARY TABLE — {mode_label}")
    print(f"  {'─'*70}")
    print(f"  {'Features':<10} {'MeanMAE':<10} {'StdMAE':<10} "
          f"{'Acc':<8} {'Spec':<8} {'F1':<8} {'Time/ep'}")
    print(f"  {'─'*70}")
    for n_f in FEATURE_SWEEP:
        r = all_results[f"{n_f}_features"]
        print(f"  {str(n_f):<10} "
              f"{r['mae']['mean_MAE']:<10} "
              f"{r['mae']['std_MAE']:<10} "
              f"{r['clf']['Accuracy']:<8} "
              f"{r['clf']['Specificity']:<8} "
              f"{r['clf']['F1']:<8} "
              f"{r['history']['avg_time_per_epoch']}s")
    print(f"  {'='*70}\n")


if __name__ == "__main__":
    main()
