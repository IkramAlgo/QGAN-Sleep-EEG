"""
plot_results_Journal.py  —  Journal-quality plots for QGAN-Sleep-EEG
=====================================================================
Reads the 5 latest JSON result files from F:\\QGAN_Project and produces:

  Fig 1 — GAN Discriminator metrics bar chart  (per experiment key)
  Fig 2 — MAE bar chart with error bars         (mean ± std across folds)
  Fig 3 — Downstream SVM / RF MacroF1 bar chart
  Fig 4 — Per-class F1 heatmap  (sleep stages W/N1/N2/N3/REM)
  Fig 5 — Cross-model comparison  (all 5 JSON files side-by-side)
  Fig 6 — Training time & params table figure

Run:  python qgan\\plot_results_Journal.py
Output: F:\\QGAN_Project\\plots\\  (PNG + PDF, 300 dpi)
"""

import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent   # qgan\ → F:\QGAN_Project
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

DPI      = 300
SAVE_PDF = True

JSON_FILES = {
    "results_simulator_spectral.json":              "QGAN Spectral",
    "results_simulator_spectral_classical_bce.json":"Classical BCE (Spectral)",
    "results_simulator_spectral_classical_wgan.json":"Classical WGAN-GP (Spectral)",
    "results_simulator_spectral_dcgan.json":        "DCGAN (Spectral)",
    "results_simulator_datanoise_statistical.json": "QGAN Noise (Statistical)",
}

# Colour per file (consistent across all figures)
FILE_COLORS = {
    "results_simulator_spectral.json":               "#4A90D9",
    "results_simulator_spectral_classical_bce.json": "#E07B54",
    "results_simulator_spectral_classical_wgan.json":"#9B59B6",
    "results_simulator_spectral_dcgan.json":         "#F1C40F",
    "results_simulator_datanoise_statistical.json":  "#6DBF82",
}

SLEEP_STAGES  = ["W", "N1", "N2", "N3", "REM"]
CLF_METRICS   = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1"]

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        120,
})

# ── HELPERS ───────────────────────────────────────────────────────────────────
def savefig(fig, name):
    p = PLOT_DIR / f"{name}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(PLOT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {p.name}")

def agg_val(agg, key):
    """Return (mean, std) from aggregated block; handles both dict and scalar."""
    v = agg.get(key, {})
    if isinstance(v, dict):
        return v.get("mean", 0), v.get("std", 0)
    return float(v), 0.0

def load_all():
    data = {}
    for fname, label in JSON_FILES.items():
        p = BASE_DIR / fname
        if p.exists():
            with open(p) as f:
                data[fname] = {"label": label, "raw": json.load(f)}
            print(f"  Loaded : {fname}")
        else:
            print(f"  MISSING: {fname}  (skipped)")
    return data

# ── FIG 1: GAN Discriminator / Classifier Metrics ────────────────────────────
def fig1_clf_metrics(all_data):
    """Bar chart: Acc/Prec/Sens/Spec/F1 mean±std for every experiment block."""
    for fname, entry in all_data.items():
        label = entry["label"]
        raw   = entry["raw"]
        tag   = fname.replace(".json","").replace("results_","")
        color = FILE_COLORS[fname]

        exp_keys = list(raw.keys())
        n_exp    = len(exp_keys)
        fig, axes = plt.subplots(1, n_exp, figsize=(5*n_exp, 5), squeeze=False)
        fig.suptitle(f"GAN Discriminator Metrics — {label}",
                     fontsize=13, fontweight="bold")

        for col, ek in enumerate(exp_keys):
            agg = raw[ek].get("aggregated", {})
            ax  = axes[0][col]
            means = [agg_val(agg, m)[0] for m in CLF_METRICS]
            stds  = [agg_val(agg, m)[1] for m in CLF_METRICS]

            bars = ax.bar(CLF_METRICS, means, yerr=stds, capsize=5,
                          color=color, alpha=0.85, edgecolor="white",
                          error_kw={"elinewidth":1.5, "ecolor":"#333"})
            for b, v, s in zip(bars, means, stds):
                ax.text(b.get_x()+b.get_width()/2, v+s+0.012,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

            ax.set_ylim(0, 1.15)
            ax.set_title(f"{ek}\n({raw[ek].get('n_features','?')} features, "
                         f"{raw[ek].get('generator_type','?')})", fontsize=9)
            ax.set_ylabel("Score" if col == 0 else "")
            ax.tick_params(axis="x", rotation=20, labelsize=9)

        fig.tight_layout()
        savefig(fig, f"fig1_clf_{tag}")

# ── FIG 2: MAE Bar Chart ──────────────────────────────────────────────────────
def fig2_mae(all_data):
    """mean_MAE and std_MAE mean±std across folds."""
    for fname, entry in all_data.items():
        label = entry["label"]
        raw   = entry["raw"]
        tag   = fname.replace(".json","").replace("results_","")
        color = FILE_COLORS[fname]

        exp_keys = list(raw.keys())
        n_exp    = len(exp_keys)
        fig, ax  = plt.subplots(figsize=(max(7, 3*n_exp), 5))
        fig.suptitle(f"MAE (mean ± std across folds) — {label}",
                     fontsize=13, fontweight="bold")

        x      = np.arange(n_exp)
        w      = 0.35
        labels = [raw[ek].get("generator_type","?") + f"\n{ek}" for ek in exp_keys]

        mean_means, mean_stds = [], []
        std_means,  std_stds  = [], []
        for ek in exp_keys:
            agg = raw[ek].get("aggregated", {})
            m, s = agg_val(agg, "mean_MAE"); mean_means.append(m); mean_stds.append(s)
            m, s = agg_val(agg, "std_MAE");  std_means.append(m);  std_stds.append(s)

        b1 = ax.bar(x - w/2, mean_means, w, yerr=mean_stds, capsize=5,
                    color=color, alpha=0.85, label="mean_MAE", edgecolor="white")
        b2 = ax.bar(x + w/2, std_means,  w, yerr=std_stds,  capsize=5,
                    color=color, alpha=0.45, label="std_MAE",  edgecolor="white")

        for bars in [b1, b2]:
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x()+b.get_width()/2, h+0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("MAE"); ax.legend()
        fig.tight_layout()
        savefig(fig, f"fig2_mae_{tag}")

# ── FIG 3: Downstream SVM / RF MacroF1 ───────────────────────────────────────
def fig3_downstream(all_data):
    """SVM and RF MacroF1 (real vs augmented) for experiments that have downstream."""
    DOWN_KEYS = [
        ("downstream_svm_real_MacroF1",  "SVM Real"),
        ("downstream_svm_aug_MacroF1",   "SVM+Aug"),
        ("downstream_rf_real_MacroF1",   "RF Real"),
        ("downstream_rf_aug_MacroF1",    "RF+Aug"),
    ]

    for fname, entry in all_data.items():
        raw = entry["raw"]
        # Check at least one exp block has downstream data
        has_ds = any(
            any(k in raw[ek].get("aggregated",{}) for k,_ in DOWN_KEYS)
            for ek in raw
        )
        if not has_ds:
            continue

        label = entry["label"]
        tag   = fname.replace(".json","").replace("results_","")
        color = FILE_COLORS[fname]
        exp_keys = list(raw.keys())
        n_exp    = len(exp_keys)

        fig, axes = plt.subplots(1, n_exp, figsize=(5*n_exp, 5), squeeze=False)
        fig.suptitle(f"Downstream Sleep-Stage Classification MacroF1 — {label}",
                     fontsize=13, fontweight="bold")

        ds_labels = [lbl for _, lbl in DOWN_KEYS]
        colors_ds = ["#4A90D9","#2ECC71","#E07B54","#F1C40F"]

        for col, ek in enumerate(exp_keys):
            agg  = raw[ek].get("aggregated", {})
            ax   = axes[0][col]
            vals = [agg_val(agg, k)[0] for k, _ in DOWN_KEYS]
            stds = [agg_val(agg, k)[1] for k, _ in DOWN_KEYS]

            bars = ax.bar(ds_labels, vals, yerr=stds, capsize=5,
                          color=colors_ds, alpha=0.85, edgecolor="white")
            for b, v, s in zip(bars, vals, stds):
                ax.text(b.get_x()+b.get_width()/2, v+s+0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

            ax.set_ylim(0, max(0.8, max(v+s for v,s in zip(vals,stds))+0.1))
            ax.set_title(f"{ek}", fontsize=9)
            ax.set_ylabel("MacroF1" if col == 0 else "")
            ax.tick_params(axis="x", rotation=15, labelsize=9)

        fig.tight_layout()
        savefig(fig, f"fig3_downstream_{tag}")

# ── FIG 4: Per-class F1 Heatmap ───────────────────────────────────────────────
def fig4_perclass_heatmap(all_data):
    """Per sleep-stage F1 heatmap averaged across folds."""
    for fname, entry in all_data.items():
        raw = entry["raw"]
        exp_keys = list(raw.keys())

        # Gather per-class F1 from folds
        fig_data = {}   # ek → {stage: [f1_fold0, f1_fold1, ...]}
        for ek in exp_keys:
            stage_f1 = {s: [] for s in SLEEP_STAGES}
            for fold in raw[ek].get("folds", []):
                ds = fold.get("downstream", {})
                for clf_key in ["svm_aug", "svm_real", "rf_aug", "rf_real"]:
                    if clf_key in ds:
                        pcf1 = ds[clf_key].get("per_class_F1", {})
                        for s in SLEEP_STAGES:
                            if s in pcf1:
                                stage_f1[s].append(pcf1[s].get("F1", 0))
                        break   # use first available classifier only
            fig_data[ek] = {s: np.mean(v) if v else 0 for s, v in stage_f1.items()}

        # Only plot if we got actual data
        if all(all(v == 0 for v in d.values()) for d in fig_data.values()):
            continue

        label = entry["label"]
        tag   = fname.replace(".json","").replace("results_","")

        matrix = np.array([[fig_data[ek].get(s, 0) for s in SLEEP_STAGES]
                            for ek in exp_keys])

        fig, ax = plt.subplots(figsize=(9, max(3, 0.7*len(exp_keys)+2)))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="F1 Score")

        ax.set_xticks(range(len(SLEEP_STAGES)))
        ax.set_xticklabels(SLEEP_STAGES, fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(exp_keys)))
        ax.set_yticklabels(exp_keys, fontsize=9)
        ax.set_title(f"Per-Class F1 (Sleep Stages) — {label}",
                     fontsize=13, fontweight="bold")

        for i in range(len(exp_keys)):
            for j, s in enumerate(SLEEP_STAGES):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if v < 0.65 else "white",
                        fontweight="bold")

        fig.tight_layout()
        savefig(fig, f"fig4_perclass_{tag}")

# ── FIG 5: Cross-model comparison ────────────────────────────────────────────
def fig5_cross_model(all_data):
    """Side-by-side bar chart comparing all loaded models on key metrics."""
    summary = {}   # label → {metric: (mean, std)}
    for fname, entry in all_data.items():
        raw   = entry["raw"]
        label = entry["label"]
        # Aggregate across all experiment blocks (there may be only one)
        all_means = {m: [] for m in CLF_METRICS + ["mean_MAE", "std_MAE"]}
        for ek in raw:
            agg = raw[ek].get("aggregated", {})
            for m in all_means:
                val, _ = agg_val(agg, m)
                if val > 0:
                    all_means[m].append(val)
        summary[label] = {
            m: (np.mean(v) if v else 0, np.std(v) if v else 0)
            for m, v in all_means.items()
        }

    models  = list(summary.keys())
    metrics = CLF_METRICS
    n_m     = len(metrics)
    n_mod   = len(models)
    colors  = list(FILE_COLORS.values())[:n_mod]

    x  = np.arange(n_m)
    w  = 0.8 / n_mod
    offsets = np.linspace(-(0.4 - w/2), 0.4 - w/2, n_mod)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (model, col) in enumerate(zip(models, colors)):
        vals = [summary[model][m][0] for m in metrics]
        stds = [summary[model][m][1] for m in metrics]
        bars = ax.bar(x + offsets[i], vals, w, yerr=stds, capsize=4,
                      color=col, alpha=0.85, label=model, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Cross-Model Comparison — All Experiments",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig5_cross_model")

# ── FIG 6: Training time & parameters ────────────────────────────────────────
def fig6_time_params(all_data):
    """Horizontal bar chart of avg_time_per_epoch and n_params_gen per model."""
    labels, times, params_gen, params_disc = [], [], [], []

    for fname, entry in all_data.items():
        raw = entry["raw"]
        for ek in raw:
            agg = raw[ek].get("aggregated", {})
            t, _ = agg_val(agg, "avg_time_per_epoch")
            pg, _ = agg_val(agg, "n_params_gen")
            pd, _ = agg_val(agg, "n_params_disc")
            if t > 0:
                labels.append(f"{entry['label']}\n{ek}")
                times.append(t)
                params_gen.append(pg)
                params_disc.append(pd)

    if not labels:
        print("  [fig6] No timing data found, skipping.")
        return

    n   = len(labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, 0.5*n+2)))
    fig.suptitle("Training Speed & Model Size", fontsize=13, fontweight="bold")

    y = np.arange(n)
    ax1.barh(y, times, color="#4A90D9", alpha=0.85, edgecolor="white")
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=7)
    ax1.set_xlabel("Avg time per epoch (s)"); ax1.invert_yaxis()
    ax1.set_title("Training Time / Epoch")
    for i, t in enumerate(times):
        ax1.text(t + max(times)*0.01, i, f"{t:.1f}s", va="center", fontsize=7)

    x2 = np.arange(len([p for p in params_gen if p > 0]))
    valid = [(l, g, d) for l, g, d in zip(labels, params_gen, params_disc) if g > 0]
    if valid:
        vlabels, vg, vd = zip(*valid)
        yv = np.arange(len(vlabels))
        ax2.barh(yv - 0.2, vg, 0.35, color="#E07B54", alpha=0.85, label="Generator")
        ax2.barh(yv + 0.2, vd, 0.35, color="#6DBF82", alpha=0.85, label="Discriminator")
        ax2.set_yticks(yv); ax2.set_yticklabels(vlabels, fontsize=7)
        ax2.set_xlabel("# Parameters"); ax2.invert_yaxis()
        ax2.set_title("Model Parameters")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, "fig6_time_params")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  QGAN-Sleep-EEG — Journal Plot Generator")
    print("="*65)
    print(f"  JSON folder : {BASE_DIR.resolve()}")
    print(f"  Plot output : {PLOT_DIR.resolve()}\n")

    all_data = load_all()
    if not all_data:
        print("\n  ERROR: No JSON files found.")
        sys.exit(1)

    print("\n--- Generating figures ---")
    fig1_clf_metrics(all_data)
    fig2_mae(all_data)
    fig3_downstream(all_data)
    fig4_perclass_heatmap(all_data)
    fig5_cross_model(all_data)
    fig6_time_params(all_data)

    print(f"\n{'='*65}")
    print(f"  Done!  Plots saved to: {PLOT_DIR.resolve()}")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()