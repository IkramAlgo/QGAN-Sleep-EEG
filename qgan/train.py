# Quantum Generative Adversarial Network (QGAN)
# Updated with evaluation metrics, classical baseline, and training fixes

import torch
import pennylane as qml
import time
import os
import numpy as np
from qgan.models import DiscriminatorQuantumCircuit, GeneratorQuantumCircuit
from qgan.classical_baseline import ClassicalGenerator, ClassicalDiscriminator
from qgan.data_loader import get_data_loader

# ── Loss ──────────────────────────────────────────────────────────────────────
criterion = torch.nn.BCELoss()

# ── Hyperparameters ───────────────────────────────────────────────────────────
epochs        = int(os.getenv("EPOCHS", "50"))   # increased from 10 → 50
batch_size    = 32                                # reduced from 64 → 32 (more stable)
learning_rate = 0.00005                           # reduced from 0.0002 → 0.00005


# ── Loss function ─────────────────────────────────────────────────────────────
def qgan_loss(generated_samples, real_samples, discriminator):
    discriminator_real_output = torch.sigmoid(discriminator(real_samples))
    discriminator_fake_output = torch.sigmoid(discriminator(generated_samples.detach()))

    generator_output = torch.sigmoid(discriminator(generated_samples))
    generator_loss = criterion(generator_output,
                               torch.ones_like(generator_output))   # generator wants to fool disc
    discriminator_loss = (
        criterion(discriminator_real_output, torch.ones_like(discriminator_real_output))
        + criterion(discriminator_fake_output, torch.zeros_like(discriminator_fake_output))
    )
    return generator_loss, discriminator_loss


# ── Evaluation metrics ────────────────────────────────────────────────────────
def evaluate_model(generator, real_data_loader, n_features, n_samples=200):
    """Compare real vs generated data distributions."""
    generator.eval()
    all_real, all_fake = [], []

    with torch.no_grad():
        for batch in real_data_loader:
            all_real.append(batch[0])
            if len(all_real) * batch_size >= n_samples:
                break
        noise = torch.randn(n_samples, n_features)
        fake  = generator(noise)
        all_fake.append(fake)

    real_tensor = torch.cat(all_real)[:n_samples]
    fake_tensor = torch.cat(all_fake)[:n_samples]

    real_mean = real_tensor.mean(dim=0)
    fake_mean = fake_tensor.mean(dim=0)
    real_std  = real_tensor.std(dim=0)
    fake_std  = fake_tensor.std(dim=0)

    # Mean Absolute Error between real and fake distributions
    mean_mae = torch.abs(real_mean - fake_mean).mean().item()
    std_mae  = torch.abs(real_std  - fake_std ).mean().item()

    generator.train()
    return {
        "real_mean": real_mean.numpy(),
        "fake_mean": fake_mean.numpy(),
        "real_std":  real_std.numpy(),
        "fake_std":  fake_std.numpy(),
        "mean_MAE":  mean_mae,
        "std_MAE":   std_mae,
    }


# ── Training loop (shared by quantum and classical) ───────────────────────────
def train_model(generator, discriminator, loader, n_features,
                label="QGAN", loss_fn=None):
    if loss_fn is None:
        loss_fn = qgan_loss

    opt_g = torch.optim.Adam(generator.parameters(),     lr=learning_rate, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # learning-rate schedulers – decay LR by 0.95 every 10 epochs
    sched_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=10, gamma=0.95)
    sched_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=10, gamma=0.95)

    history = {"gen_loss": [], "disc_loss": [], "mean_MAE": [], "std_MAE": []}

    print(f"\n{'='*60}")
    print(f"  Training {label}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        epoch_start = time.time()
        gen_losses, disc_losses = [], []

        for batch in loader:
            real_samples = batch[0]
            current_batch = real_samples.shape[0]

            # ── Generator update ──────────────────────────────────────────
            noise            = torch.randn(current_batch, n_features)
            generated        = generator(noise)
            gen_loss, _      = loss_fn(generated, real_samples, discriminator)

            opt_g.zero_grad()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            opt_g.step()

            # ── Discriminator update ──────────────────────────────────────
            noise      = torch.randn(current_batch, n_features)
            generated  = generator(noise)
            _, disc_loss = loss_fn(generated, real_samples, discriminator)

            opt_d.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            opt_d.step()

            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())

        sched_g.step()
        sched_d.step()

        avg_g = np.mean(gen_losses)
        avg_d = np.mean(disc_losses)
        elapsed = time.time() - epoch_start

        # evaluate every 10 epochs
        metrics = {}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            metrics = evaluate_model(generator, loader, n_features)
            history["mean_MAE"].append(metrics["mean_MAE"])
            history["std_MAE"].append(metrics["std_MAE"])
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f} | "
                  f"Mean MAE: {metrics['mean_MAE']:.4f} | "
                  f"Std  MAE: {metrics['std_MAE']:.4f} | "
                  f"Time: {elapsed:.1f}s")
        else:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f} | "
                  f"Time: {elapsed:.1f}s")

        history["gen_loss"].append(avg_g)
        history["disc_loss"].append(avg_d)

    return history


# ── Final comparison report ───────────────────────────────────────────────────
def print_comparison(q_history, c_history):
    print(f"\n{'='*60}")
    print("  QGAN vs Classical GAN - Final Comparison")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'QGAN':>12} {'Classical':>12}")
    print(f"{'-'*50}")

    q_mae  = q_history["mean_MAE"][-1]  if q_history["mean_MAE"]  else float("nan")
    c_mae  = c_history["mean_MAE"][-1]  if c_history["mean_MAE"]  else float("nan")
    q_smae = q_history["std_MAE"][-1]   if q_history["std_MAE"]   else float("nan")
    c_smae = c_history["std_MAE"][-1]   if c_history["std_MAE"]   else float("nan")
    q_gl   = q_history["gen_loss"][-1]
    c_gl   = c_history["gen_loss"][-1]

    print(f"{'Final Gen Loss':<25} {q_gl:>12.4f} {c_gl:>12.4f}")
    print(f"{'Mean MAE (lower=better)':<25} {q_mae:>12.4f} {c_mae:>12.4f}")
    print(f"{'Std  MAE (lower=better)':<25} {q_smae:>12.4f} {c_smae:>12.4f}")
    print(f"{'='*60}")

    winner_mean = "QGAN" if q_mae < c_mae else "Classical"
    winner_std  = "QGAN" if q_smae < c_smae else "Classical"
    print(f"  Better mean distribution match : {winner_mean}")
    print(f"  Better std  distribution match : {winner_std}")
    print(f"{'='*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    loader       = get_data_loader(batch_size)
    sample_batch = next(iter(loader))[0]
    n_features   = sample_batch.shape[1]

    print(f"\nDataset loaded | n_features = {n_features} | batch_size = {batch_size}")
    print(f"Epochs = {epochs} | Learning rate = {learning_rate}")

    # ── Quantum GAN ───────────────────────────────────────────────────────────
    q_gen  = GeneratorQuantumCircuit(n_qubits=n_features)
    q_disc = DiscriminatorQuantumCircuit(n_qubits=n_features)
    q_hist = train_model(q_gen, q_disc, loader, n_features, label="Quantum GAN")

    # ── Classical GAN baseline ────────────────────────────────────────────────
    c_gen  = ClassicalGenerator(n_features)
    c_disc = ClassicalDiscriminator(n_features)

    def classical_loss(generated, real, discriminator):
        d_real = torch.sigmoid(discriminator(real))
        d_fake = torch.sigmoid(discriminator(generated.detach()))
        g_out  = torch.sigmoid(discriminator(generated))
        g_loss = criterion(g_out, torch.ones_like(g_out))
        d_loss = (criterion(d_real, torch.ones_like(d_real))
                  + criterion(d_fake, torch.zeros_like(d_fake)))
        return g_loss, d_loss

    c_hist = train_model(c_gen, c_disc, loader, n_features,
                         label="Classical GAN", loss_fn=classical_loss)

    # ── Comparison ────────────────────────────────────────────────────────────
    print_comparison(q_hist, c_hist)


if __name__ == "__main__":
    main()