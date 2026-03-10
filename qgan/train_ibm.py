# qgan/train_ibm.py
# Trains QGAN with IBM 127-qubit noise simulation
# Optimized for speed — limits samples per epoch
# Run: python -m qgan.train_ibm

import torch, copy, time, json, os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

from qgan.config      import LEARNING_RATE, GRAD_CLIP, EVAL_SAMPLES
from qgan.models_ibm  import GeneratorIBM, DiscriminatorIBM
from qgan.data_loader import load_sleep_edf

N_FEATURES       = 4
EPOCHS_IBM       = int(os.getenv("EPOCHS_IBM",    "10"))
BATCH_IBM        = int(os.getenv("BATCH_IBM",     "4"))
SHOTS            = int(os.getenv("SHOTS",         "512"))
# KEY: only use this many samples per epoch — makes each epoch fast
SAMPLES_PER_EPOCH = int(os.getenv("SAMPLES_PER_EPOCH", "32"))

BCE = torch.nn.BCELoss()


def load_data():
    data   = load_sleep_edf()[:, :N_FEATURES]
    # use only SAMPLES_PER_EPOCH samples per epoch for speed
    subset = data[:SAMPLES_PER_EPOCH]
    loader = DataLoader(TensorDataset(subset),
                        batch_size=BATCH_IBM, shuffle=True, drop_last=True)
    return loader, data   # full data for eval, subset loader for training


def compute_mae(generator, data):
    generator.eval()
    with torch.no_grad():
        real = data[:100]
        fake = generator(torch.randn(100, N_FEATURES))
    generator.train()
    return {
        "mean_MAE": round(torch.abs(real.mean(0)-fake.mean(0)).mean().item(), 4),
        "std_MAE":  round(torch.abs(real.std(0) -fake.std(0)).mean().item(),  4),
    }


def compute_clf(generator, disc, data):
    generator.eval(); disc.eval()
    n = min(50, len(data))
    with torch.no_grad():
        real = data[:n]
        fake = generator(torch.randn(n, N_FEATURES))
        rs   = torch.sigmoid(disc(real))
        fs   = torch.sigmoid(disc(fake))
        rs   = rs.mean(1) if rs.dim() > 1 else rs
        fs   = fs.mean(1) if fs.dim() > 1 else fs
    scores = torch.cat([rs, fs]).numpy()
    labels = np.array([1]*n + [0]*n)
    preds  = (scores > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    generator.train(); disc.train()
    return {
        "Accuracy":    round(accuracy_score(labels, preds), 4),
        "Precision":   round(precision_score(labels, preds, zero_division=0), 4),
        "Sensitivity": round(recall_score(labels, preds, zero_division=0), 4),
        "Specificity": round(tn/(tn+fp) if (tn+fp)>0 else 0.0, 4),
        "F1":          round(f1_score(labels, preds, zero_division=0), 4),
    }


def train(generator, disc, loader, data):
    opt_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5,0.999))
    opt_d = torch.optim.Adam(disc.parameters(),      lr=LEARNING_RATE, betas=(0.5,0.999))

    history = {"gen_loss":[],"disc_loss":[],"mean_MAE":[],"std_MAE":[],"mae_epochs":[],"times":[]}
    best_d  = float("inf")
    best_gs = copy.deepcopy(generator.state_dict())
    best_ds = copy.deepcopy(disc.state_dict())

    batches_per_epoch = len(loader)
    print(f"\n  ╔════════════════════════════════════════════════════════╗")
    print(f"  ║  QGAN QPU-Sim Training                                ║")
    print(f"  ║  Device          : {generator.device_label:<36}║")
    print(f"  ║  Epochs          : {EPOCHS_IBM:<36}║")
    print(f"  ║  Samples/epoch   : {SAMPLES_PER_EPOCH:<36}║")
    print(f"  ║  Batch size      : {BATCH_IBM:<36}║")
    print(f"  ║  Batches/epoch   : {batches_per_epoch:<36}║")
    print(f"  ║  Shots           : {SHOTS:<36}║")
    print(f"  ║  Est. time/epoch : {str(batches_per_epoch*2*30)+'s approx':<36}║")
    print(f"  ╚════════════════════════════════════════════════════════╝\n")

    for epoch in range(EPOCHS_IBM):
        t0 = time.time()
        gl, dl = [], []

        for batch in loader:
            real = batch[0]; bs = real.shape[0]

            # generator step
            fake   = generator(torch.randn(bs, N_FEATURES))
            pf_g   = torch.sigmoid(disc(fake))
            g_loss = BCE(pf_g, torch.ones_like(pf_g))
            opt_g.zero_grad(); g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
            opt_g.step()

            # discriminator step
            fake   = generator(torch.randn(bs, N_FEATURES)).detach()
            pr     = torch.sigmoid(disc(real))
            pf     = torch.sigmoid(disc(fake))
            d_loss = BCE(pr, torch.ones_like(pr)) + BCE(pf, torch.zeros_like(pf))
            opt_d.zero_grad(); d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), GRAD_CLIP)
            opt_d.step()

            gl.append(g_loss.item()); dl.append(d_loss.item())

        avg_g   = float(np.mean(gl))
        avg_d   = float(np.mean(dl))
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
        history["mae_epochs"].append(epoch+1)
        print(f"  Epoch [{epoch+1:3d}/{EPOCHS_IBM}] "
              f"G:{avg_g:.4f} D:{avg_d:.4f} "
              f"MeanMAE:{mae['mean_MAE']:.4f} "
              f"StdMAE:{mae['std_MAE']:.4f} "
              f"Time:{elapsed:.1f}s")

    generator.load_state_dict(best_gs)
    disc.load_state_dict(best_ds)
    history["avg_time"] = round(float(np.mean(history["times"])), 4)
    return history, generator, disc


def print_table(ibm):
    try:
        with open("results.json") as f:
            cpu_all = json.load(f)
        cpu = next(r for r in cpu_all if r["n_features"] == N_FEATURES)
    except Exception:
        print("  results.json not found — run python -m qgan.train first")
        return

    c  = cpu["classical"]
    qc = cpu["qgan"]
    qi = ibm

    print(f"\n{'═'*82}")
    print(f"  COMPLETE RESULTS — Classical GAN / QGAN CPU-Sim / QGAN QPU-Sim")
    print(f"  Sleep EEG | {N_FEATURES} features | CPU Epochs=50, QPU Epochs={EPOCHS_IBM}")
    print(f"{'═'*82}")
    fmt = "  {:<18} {:<22} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9}"
    print(fmt.format("Model","Hardware","Acc","Prec","Sens","Spec","F1","Time/ep"))
    print(f"  {'-'*80}")
    print(fmt.format("Classical GAN","CPU",
        c["clf"]["Accuracy"], c["clf"]["Precision"],
        c["clf"]["Sensitivity"], c["clf"]["Specificity"],
        c["clf"]["F1"], f"{c['history']['avg_time']:.2f}s"))
    print(fmt.format("QGAN","CPU-Sim (noiseless)",
        qc["clf"]["Accuracy"], qc["clf"]["Precision"],
        qc["clf"]["Sensitivity"], qc["clf"]["Specificity"],
        qc["clf"]["F1"], f"{qc['history']['avg_time']:.2f}s"))
    print(fmt.format("QGAN","QPU-Sim (IBM 127q)",
        qi["clf"]["Accuracy"], qi["clf"]["Precision"],
        qi["clf"]["Sensitivity"], qi["clf"]["Specificity"],
        qi["clf"]["F1"], f"{qi['history']['avg_time']:.2f}s"))
    print(f"{'═'*82}\n")


def main():
    print(f"\n{'='*58}")
    print(f"  QGAN QPU-Sim — IBM 127-qubit noise model")
    print(f"  Epochs={EPOCHS_IBM}  Batch={BATCH_IBM}  Shots={SHOTS}")
    print(f"  Samples per epoch={SAMPLES_PER_EPOCH} (speed optimization)")
    print(f"{'='*58}")

    loader, data = load_data()

    print("\n  Building QPU-Sim models...")
    q_gen  = GeneratorIBM(n_qubits=N_FEATURES, shots=SHOTS)
    q_disc = DiscriminatorIBM(n_qubits=N_FEATURES, shots=SHOTS)

    history, q_gen, q_disc = train(q_gen, q_disc, loader, data)

    print("\n  Computing final classification metrics...")
    clf = compute_clf(q_gen, q_disc, data)
    print("\n  QPU-Sim Results:")
    for k, v in clf.items():
        print(f"    {k:<15}: {v}")

    ibm_results = {
        "n_features":        N_FEATURES,
        "device":            q_gen.device_label,
        "shots":             SHOTS,
        "epochs":            EPOCHS_IBM,
        "samples_per_epoch": SAMPLES_PER_EPOCH,
        "history":           history,
        "clf":               clf,
    }
    with open("results_ibm.json","w") as f:
        json.dump(ibm_results, f, indent=2)
    print(f"\n  Saved: results_ibm.json")
    print_table(ibm_results)
    print("  Next step: python -m qgan.plot_final_ibm")


if __name__ == "__main__":
    main()         