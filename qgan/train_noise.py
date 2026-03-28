# qgan/train_noise.py
# Noise Experiment — 3-Model Comparison
# Compares: Classical GAN | QGAN Noiseless | QGAN Noisy Data
#
# Run: python -m qgan.train_noise
#
# Noise is added to EEG training data following the QCNN Qiskit
# generate_dataset approach — Gaussian noise injected into features,
# clamped to [-1, 1] to match quantum circuit output range.
#
# Architecture: Arch C (6 qubits, ring CNOT, RX->CNOT->RY, WGAN-GP)
# Feature sweep: [2, 3, 4]
# Output: results_noise.json

import copy
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

from qgan.config      import LEARNING_RATE, GRAD_CLIP
from qgan.data_loader import load_sleep_edf
from qgan.models_noise import (
    ClassicalGenerator,
    GeneratorArchC,
    ClassicalDiscriminator,
    add_data_noise,
)

# ================================================================
#  CONFIG — tune here for your machine
# ================================================================
FEATURE_SWEEP = [2, 3, 4]
FEATURE_NAMES = {
    2: ["Mean", "Std Dev"],
    3: ["Mean", "Std Dev", "Min"],
    4: ["Mean", "Std Dev", "Min", "Max"],
}
N_QUBITS    = 6       # Arch C: always 6 qubits
N_LAYERS    = 2
BATCH_SIZE  = 16      # larger batch = faster epochs on CPU
LAMBDA_GP   = 10      # WGAN-GP gradient penalty weight
NOISE_LEVEL = 0.1     # 10% Gaussian noise — same as QCNN tutorial default
EPOCHS      = 50      # target — reduce if too slow on your machine
OUT_FILE    = "results_noise.json"

# Learning rates
LR_CLASSICAL_G = LEARNING_RATE
LR_CLASSICAL_D = LEARNING_RATE * 5.0
LR_QUANTUM_G   = LEARNING_RATE
LR_QUANTUM_D   = LEARNING_RATE * 5.0
# ================================================================


# ================================================================
#  LOSS FUNCTIONS
# ================================================================
BCE = torch.nn.BCELoss()


def bce_loss(fake, real, disc):
    """Standard GAN loss for Classical GAN."""
    pred_real         = torch.sigmoid(disc(real))
    pred_fake_detach  = torch.sigmoid(disc(fake.detach()))
    pred_fake_for_gen = torch.sigmoid(disc(fake))

    g_loss = BCE(pred_fake_for_gen, torch.ones_like(pred_fake_for_gen))
    d_loss = (BCE(pred_real,       torch.ones_like(pred_real)) +
              BCE(pred_fake_detach, torch.zeros_like(pred_fake_detach)))
    return g_loss, d_loss


def gradient_penalty(disc, real, fake):
    """WGAN-GP gradient penalty."""
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
def compute_mae(generator, data, n_features, is_quantum):
    generator.eval()
    with torch.no_grad():
        real = data[:100, :n_features].float()
        z    = torch.randn(100, n_features)

        if is_quantum:
            fake_all = generator(z)
            fake     = fake_all[:, :n_features]
        else:
            fake = generator(z)[:, :n_features]

    generator.train()
    return {
        "mean_MAE": round(torch.abs(real.mean(0) - fake.mean(0)).mean().item(), 4),
        "std_MAE":  round(torch.abs(real.std(0)  - fake.std(0) ).mean().item(), 4),
    }


def compute_clf(generator, disc, data, n_features, is_quantum, use_wgan):
    generator.eval(); disc.eval()
    n = min(100, len(data))
    with torch.no_grad():
        real = data[:n, :n_features].float()
        z    = torch.randn(n, n_features)

        if is_quantum:
            fake = generator(z)
        else:
            fake = generator(z)

        # Pad real to N_QUBITS for discriminator
        if real.shape[-1] < N_QUBITS:
            pad     = torch.zeros(n, N_QUBITS - real.shape[-1])
            real_in = torch.cat([real, pad], dim=-1)
        else:
            real_in = real

        if use_wgan:
            rs = disc(real_in).squeeze()
            fs = disc(fake).squeeze()
            threshold = 0.0
        else:
            rs = torch.sigmoid(disc(real_in)).squeeze()
            fs = torch.sigmoid(disc(fake)).squeeze()
            threshold = 0.5

    scores = torch.cat([rs, fs]).detach().numpy()
    labels = np.array([1]*n + [0]*n)
    preds  = (scores > threshold).astype(int)

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
#  CLASSICAL GAN TRAINING — BCE loss
# ================================================================
def train_classical(generator, disc, loader, data, n_features, n_epochs, label):
    opt_g = torch.optim.Adam(generator.parameters(),
                              lr=LR_CLASSICAL_G, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(),
                              lr=LR_CLASSICAL_D, betas=(0.5, 0.999))

    history = {"gen_loss": [], "disc_loss": [], "mean_MAE": [],
               "std_MAE": [], "mae_epochs": [], "times": []}
    best_d  = float("inf")
    best_gs = copy.deepcopy(generator.state_dict())
    best_ds = copy.deepcopy(disc.state_dict())

    for epoch in range(n_epochs):
        t0 = time.time()
        g_losses, d_losses = [], []

        for batch in loader:
            real = batch[0].float()
            bs   = real.shape[0]

            # Pad real to N_QUBITS for discriminator
            if real.shape[-1] < N_QUBITS:
                pad         = torch.zeros(bs, N_QUBITS - real.shape[-1])
                real_padded = torch.cat([real, pad], dim=-1)
            else:
                real_padded = real

            z    = torch.randn(bs, n_features)
            fake = generator(z)

            g_loss, d_loss = bce_loss(fake, real_padded, disc)

            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            fake = generator(torch.randn(bs, n_features))
            _, d_loss = bce_loss(fake, real_padded, disc)
            opt_d.zero_grad(); d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), GRAD_CLIP)
            opt_d.step()

            g_losses.append(g_loss.item())
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

        mae = compute_mae(generator, data, n_features, is_quantum=False)
        history["mean_MAE"].append(mae["mean_MAE"])
        history["std_MAE"].append(mae["std_MAE"])
        history["mae_epochs"].append(epoch + 1)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == n_epochs:
            print(f"    [{label}] Epoch [{epoch+1:3d}/{n_epochs}] "
                  f"G:{avg_g:.4f} D:{avg_d:.4f} "
                  f"MeanMAE:{mae['mean_MAE']:.4f} StdMAE:{mae['std_MAE']:.4f} "
                  f"Time:{elapsed:.1f}s")

    generator.load_state_dict(best_gs)
    disc.load_state_dict(best_ds)
    history["avg_time_per_epoch"] = round(float(np.mean(history["times"])), 2)
    return history, generator, disc


# ================================================================
#  QGAN TRAINING — WGAN-GP loss
#  Works for both noiseless and noisy data — pass noisy loader for noise exp
# ================================================================
def train_qgan(generator, disc, loader, data, n_features, n_epochs, label):
    # WGAN-GP Adam betas: (0.0, 0.9)
    opt_g = torch.optim.Adam(generator.parameters(),
                              lr=LR_QUANTUM_G, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(),
                              lr=LR_QUANTUM_D, betas=(0.0, 0.9))

    history = {"gen_loss": [], "disc_loss": [], "mean_MAE": [],
               "std_MAE": [], "mae_epochs": [], "times": []}
    best_d  = float("inf")
    best_gs = copy.deepcopy(generator.state_dict())
    best_ds = copy.deepcopy(disc.state_dict())

    for epoch in range(n_epochs):
        t0 = time.time()
        g_losses, d_losses = [], []

        for batch in loader:
            real = batch[0].float()
            bs   = real.shape[0]

            # Pad real to N_QUBITS for discriminator
            if real.shape[-1] < N_QUBITS:
                pad         = torch.zeros(bs, N_QUBITS - real.shape[-1])
                real_padded = torch.cat([real, pad], dim=-1)
            else:
                real_padded = real

            # Generator — 2 steps per discriminator (faster than 3, still stable)
            for _ in range(2):
                fake   = generator(torch.randn(bs, n_features))
                g_loss = -disc(fake).mean()
                opt_g.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
                opt_g.step()
                g_losses.append(g_loss.item())

            # Discriminator — 1 step with gradient penalty
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

        mae = compute_mae(generator, data, n_features, is_quantum=True)
        history["mean_MAE"].append(mae["mean_MAE"])
        history["std_MAE"].append(mae["std_MAE"])
        history["mae_epochs"].append(epoch + 1)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == n_epochs:
            print(f"    [{label}] Epoch [{epoch+1:3d}/{n_epochs}] "
                  f"G:{avg_g:+.4f} C:{avg_d:+.4f} "
                  f"MeanMAE:{mae['mean_MAE']:.4f} StdMAE:{mae['std_MAE']:.4f} "
                  f"Time:{elapsed:.1f}s")

    generator.load_state_dict(best_gs)
    disc.load_state_dict(best_ds)
    history["avg_time_per_epoch"] = round(float(np.mean(history["times"])), 2)
    return history, generator, disc


# ================================================================
#  MAIN
# ================================================================
def main():
    print(f"\n  {'='*65}")
    print(f"  NOISE EXPERIMENT — 3-Model Comparison")
    print(f"  Models  : Classical GAN | QGAN Noiseless | QGAN Noisy")
    print(f"  Arch    : Arch C — 6 qubits | ring CNOT | RX->CNOT->RY")
    print(f"  Loss    : Classical=BCE | QGAN=WGAN-GP")
    print(f"  Noise   : Gaussian {NOISE_LEVEL} (QCNN Qiskit style)")
    print(f"  Features: {FEATURE_SWEEP}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  {'='*65}\n")

    all_data = load_sleep_edf()
    all_results = {}

    for n_features in FEATURE_SWEEP:
        feat_names = FEATURE_NAMES[n_features]

        print(f"\n  {'='*65}")
        print(f"  EXPERIMENT: {n_features} features {feat_names}")
        print(f"  {'='*65}")

        # Clean data loader — for Classical GAN and QGAN Noiseless
        clean_data   = all_data[:, :n_features]
        clean_loader = DataLoader(TensorDataset(clean_data),
                                  batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=True)

        # Noisy data loader — for QGAN Noisy
        # Noise applied per batch inside loader for variety each epoch
        noisy_data   = add_data_noise(clean_data, noise_level=NOISE_LEVEL)
        noisy_loader = DataLoader(TensorDataset(noisy_data),
                                  batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=True)

        print(f"  Clean data : {len(clean_data)} samples")
        print(f"  Noisy data : same {len(noisy_data)} samples + "
              f"N(0, {NOISE_LEVEL}) noise, clamped to [-1, 1]")
        print(f"  Noise mean shift: "
              f"{(noisy_data - clean_data).abs().mean().item():.4f}")

        # ── 1. CLASSICAL GAN ─────────────────────────────────────
        print(f"\n  [1/3] Classical GAN (BCE)")
        c_gen  = ClassicalGenerator(latent_dim=n_features, output_dim=N_QUBITS)
        c_disc = ClassicalDiscriminator(input_dim=N_QUBITS)

        c_hist, c_gen, c_disc = train_classical(
            c_gen, c_disc, clean_loader, clean_data,
            n_features, EPOCHS, "Classical GAN"
        )
        c_clf = compute_clf(c_gen, c_disc, clean_data, n_features,
                            is_quantum=False, use_wgan=False)
        c_mae = compute_mae(c_gen, clean_data, n_features, is_quantum=False)
        print(f"  Classical GAN → Acc:{c_clf['Accuracy']} "
              f"Spec:{c_clf['Specificity']} F1:{c_clf['F1']} "
              f"StdMAE:{c_mae['std_MAE']}")

        # ── 2. QGAN NOISELESS ────────────────────────────────────
        print(f"\n  [2/3] QGAN Noiseless (Arch C + WGAN-GP)")
        q_gen_clean  = GeneratorArchC(n_qubits=N_QUBITS, n_layers=N_LAYERS)
        q_disc_clean = ClassicalDiscriminator(input_dim=N_QUBITS)

        qc_hist, q_gen_clean, q_disc_clean = train_qgan(
            q_gen_clean, q_disc_clean, clean_loader, clean_data,
            n_features, EPOCHS, "QGAN Noiseless"
        )
        qc_clf = compute_clf(q_gen_clean, q_disc_clean, clean_data, n_features,
                             is_quantum=True, use_wgan=True)
        qc_mae = compute_mae(q_gen_clean, clean_data, n_features, is_quantum=True)
        print(f"  QGAN Noiseless → Acc:{qc_clf['Accuracy']} "
              f"Spec:{qc_clf['Specificity']} F1:{qc_clf['F1']} "
              f"StdMAE:{qc_mae['std_MAE']}")

        # ── 3. QGAN NOISY DATA ───────────────────────────────────
        print(f"\n  [3/3] QGAN Noisy Data (Arch C + WGAN-GP + noise)")
        q_gen_noisy  = GeneratorArchC(n_qubits=N_QUBITS, n_layers=N_LAYERS)
        q_disc_noisy = ClassicalDiscriminator(input_dim=N_QUBITS)

        qn_hist, q_gen_noisy, q_disc_noisy = train_qgan(
            q_gen_noisy, q_disc_noisy, noisy_loader, clean_data,
            n_features, EPOCHS, "QGAN Noisy"
        )
        # Evaluate on CLEAN data — measure how well noisy training generalizes
        qn_clf = compute_clf(q_gen_noisy, q_disc_noisy, clean_data, n_features,
                             is_quantum=True, use_wgan=True)
        qn_mae = compute_mae(q_gen_noisy, clean_data, n_features, is_quantum=True)
        print(f"  QGAN Noisy → Acc:{qn_clf['Accuracy']} "
              f"Spec:{qn_clf['Specificity']} F1:{qn_clf['F1']} "
              f"StdMAE:{qn_mae['std_MAE']}")

        # ── Save results for this feature count ──────────────────
        all_results[f"{n_features}_features"] = {
            "n_features":    n_features,
            "feature_names": feat_names,
            "noise_level":   NOISE_LEVEL,
            "epochs":        EPOCHS,

            "classical_gan": {
                "loss":    "BCE",
                "history": c_hist,
                "mae":     c_mae,
                "clf":     c_clf,
            },
            "qgan_noiseless": {
                "loss":    "WGAN-GP",
                "architecture": "Arch C — 6-qubit ring CNOT RX->CNOT->RY",
                "history": qc_hist,
                "mae":     qc_mae,
                "clf":     qc_clf,
            },
            "qgan_noisy": {
                "loss":       "WGAN-GP",
                "architecture": "Arch C — 6-qubit ring CNOT RX->CNOT->RY",
                "noise_level": NOISE_LEVEL,
                "history":    qn_hist,
                "mae":        qn_mae,
                "clf":        qn_clf,
                "note": "Trained on noisy data, evaluated on clean data",
            },
        }

        # Save per-feature immediately — do not lose progress
        per_file = OUT_FILE.replace(".json", f"_{n_features}f.json")
        with open(per_file, "w") as f:
            json.dump(all_results[f"{n_features}_features"], f, indent=2)
        print(f"\n  Saved: {per_file}")

    # Save combined results
    with open(OUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final summary table
    print(f"\n  {'='*75}")
    print(f"  FINAL SUMMARY TABLE — Noise Experiment")
    print(f"  {'─'*75}")
    print(f"  {'Features':<10} {'Model':<20} {'Acc':<8} {'Spec':<8} "
          f"{'F1':<8} {'StdMAE':<10} {'Time/ep'}")
    print(f"  {'─'*75}")
    for n_f in FEATURE_SWEEP:
        r = all_results[f"{n_f}_features"]
        for model_key, label in [
            ("classical_gan",   "Classical GAN  "),
            ("qgan_noiseless",  "QGAN Noiseless "),
            ("qgan_noisy",      "QGAN Noisy     "),
        ]:
            m = r[model_key]
            print(f"  {str(n_f):<10} {label:<20} "
                  f"{m['clf']['Accuracy']:<8} "
                  f"{m['clf']['Specificity']:<8} "
                  f"{m['clf']['F1']:<8} "
                  f"{m['mae']['std_MAE']:<10} "
                  f"{m['history']['avg_time_per_epoch']}s")
        print(f"  {'─'*75}")
    print(f"  Combined results saved: {OUT_FILE}")
    print(f"  {'='*75}\n")


if __name__ == "__main__":
    main()