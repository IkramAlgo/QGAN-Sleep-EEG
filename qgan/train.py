# qgan/train.py
# Runs Hybrid QGAN (Quantum Generator + Classical Discriminator) vs Classical GAN
# Architecture 1: replaces quantum discriminator with classical MLP
# Saves all results to results.json for plotting
# Run: python -m qgan.train

import torch
import copy
import time
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from qgan.config import (EPOCHS, BATCH_SIZE, LEARNING_RATE, LR_STEP_SIZE,
                          LR_GAMMA, GRAD_CLIP, EVAL_EVERY, EVAL_SAMPLES,
                          FEATURE_SWEEP, ALL_FEATURE_NAMES, WEIGHT_INIT_STD)

# CHANGED: removed DiscriminatorQuantumCircuit, added ClassicalDiscriminator
from qgan.models import GeneratorQuantumCircuit, ClassicalDiscriminator

from qgan.classical_baseline import ClassicalGenerator, ClassicalDiscriminator as ClassicalDiscriminatorBaseline
from qgan.data_loader import get_data_loader

BCE = torch.nn.BCELoss()


def gan_loss(fake, real, disc):
    pred_real         = torch.sigmoid(disc(real))
    pred_fake         = torch.sigmoid(disc(fake.detach()))
    pred_fake_for_gen = torch.sigmoid(disc(fake))
    g_loss = BCE(pred_fake_for_gen, torch.ones_like(pred_fake_for_gen))
    d_loss = (BCE(pred_real, torch.ones_like(pred_real)) +
              BCE(pred_fake, torch.zeros_like(pred_fake)))
    return g_loss, d_loss


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


def classification_metrics(generator, disc, loader, n_features):
    generator.eval()
    disc.eval()
    real_batches = []
    with torch.no_grad():
        for batch in loader:
            real_batches.append(batch[0])
            if len(real_batches) * BATCH_SIZE >= EVAL_SAMPLES:
                break
        real = torch.cat(real_batches)[:EVAL_SAMPLES]
        fake = generator(torch.randn(len(real), n_features))
        rs = torch.sigmoid(disc(real))
        fs = torch.sigmoid(disc(fake))
        rs = rs.mean(1) if rs.dim() > 1 else rs
        fs = fs.mean(1) if fs.dim() > 1 else fs

    scores = torch.cat([rs, fs]).numpy()
    labels = np.array([1] * len(real) + [0] * len(fake))
    preds  = (scores > 0.5).astype(int)
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


def train(generator, disc, loader, n_features, name):
    # discriminator learns 5x faster than generator
    # reason: classical discriminator needs to stay ahead of quantum generator
    #         so it can provide meaningful gradients back to generator
    opt_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE,       betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(),      lr=LEARNING_RATE * 5.0, betas=(0.5, 0.999))
    sch_g = torch.optim.lr_scheduler.StepLR(opt_g, LR_STEP_SIZE, LR_GAMMA)
    sch_d = torch.optim.lr_scheduler.StepLR(opt_d, LR_STEP_SIZE, LR_GAMMA)

    history = {"gen_loss": [], "disc_loss": [], "mean_MAE": [],
               "std_MAE": [], "mae_epochs": [], "times": []}

    best_disc_loss  = float("inf")
    best_disc_state = copy.deepcopy(disc.state_dict())
    best_gen_state  = copy.deepcopy(generator.state_dict())

    print(f"\n  [{name}]")

    for epoch in range(EPOCHS):
        t0 = time.time()
        g_losses, d_losses = [], []

        for batch in loader:
            real = batch[0]
            bs   = real.shape[0]

            fake      = generator(torch.randn(bs, n_features))
            g_loss, _ = gan_loss(fake, real, disc)
            opt_g.zero_grad(); g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
            opt_g.step()

            fake       = generator(torch.randn(bs, n_features))
            _, d_loss  = gan_loss(fake, real, disc)
            opt_d.zero_grad(); d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), GRAD_CLIP)
            opt_d.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        sch_g.step(); sch_d.step()

        avg_g   = np.mean(g_losses)
        avg_d   = np.mean(d_losses)
        elapsed = time.time() - t0

        history["gen_loss"].append(avg_g)
        history["disc_loss"].append(avg_d)
        history["times"].append(elapsed)

        if avg_d < best_disc_loss:
            best_disc_loss  = avg_d
            best_disc_state = copy.deepcopy(disc.state_dict())
            best_gen_state  = copy.deepcopy(generator.state_dict())

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

    disc.load_state_dict(best_disc_state)
    generator.load_state_dict(best_gen_state)
    history["avg_time"] = round(float(np.mean(history["times"])), 4)
    return history, generator, disc


def run_experiment(n_features):
    """Run one full experiment for a given number of features."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {n_features} features "
          f"({ALL_FEATURE_NAMES[:n_features]})")
    print(f"{'='*60}")

    loader = get_data_loader(n_features)

    # CHANGED: q_disc is now ClassicalDiscriminator instead of DiscriminatorQuantumCircuit
    # Generator stays quantum — variance advantage preserved
    # Discriminator is now classical — no barren plateau, gradients flow
    q_gen  = GeneratorQuantumCircuit(n_qubits=n_features)
    q_disc = ClassicalDiscriminator(input_dim=n_features)
    q_hist, q_gen, q_disc = train(q_gen, q_disc, loader, n_features, "Hybrid QGAN (Quantum Gen + Classical Disc)")

    # Classical GAN — fully classical, unchanged
    c_gen  = ClassicalGenerator(n_features)
    c_disc = ClassicalDiscriminatorBaseline(n_features)
    c_hist, c_gen, c_disc = train(c_gen, c_disc, loader, n_features, "Classical GAN")

    # Classification metrics
    q_clf = classification_metrics(q_gen, q_disc, loader, n_features)
    c_clf = classification_metrics(c_gen, c_disc, loader, n_features)

    print(f"\n  Results ({n_features} features):")
    print(f"  HybridQGAN — MeanMAE:{q_hist['mean_MAE'][-1]:.4f} StdMAE:{q_hist['std_MAE'][-1]:.4f} "
          f"Acc:{q_clf['Accuracy']:.4f} F1:{q_clf['F1']:.4f} "
          f"Spec:{q_clf['Specificity']:.4f} AvgTime:{q_hist['avg_time']:.1f}s")
    print(f"  Classical  — MeanMAE:{c_hist['mean_MAE'][-1]:.4f} StdMAE:{c_hist['std_MAE'][-1]:.4f} "
          f"Acc:{c_clf['Accuracy']:.4f} F1:{c_clf['F1']:.4f} "
          f"Spec:{c_clf['Specificity']:.4f} AvgTime:{c_hist['avg_time']:.1f}s")

    return {
        "n_features":    n_features,
        "feature_names": ALL_FEATURE_NAMES[:n_features],
        "qgan": {
            "history": {k: [float(v) for v in vals] if isinstance(vals, list) else float(vals)
                        for k, vals in q_hist.items()},
            "clf":     q_clf,
        },
        "classical": {
            "history": {k: [float(v) for v in vals] if isinstance(vals, list) else float(vals)
                        for k, vals in c_hist.items()},
            "clf":     c_clf,
        }
    }


def main():
    all_results = []

    for n_features in FEATURE_SWEEP:
        result = run_experiment(n_features)
        all_results.append(result)

    # save results for plotting
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n  Saved results to results.json")
    print("  Run: python -m qgan.plot  to generate all paper figures")


if __name__ == "__main__":
    main()