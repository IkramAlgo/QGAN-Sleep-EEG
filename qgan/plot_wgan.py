# qgan/plot_wgan.py
# Generates figures from results_wgan.json (WGAN-GP experiment)
# Also generates the key comparison figure: BCE vs WGAN-GP side by side
# Run: python -m qgan.plot_wgan

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from qgan.config import FIGURES_DIR, EPOCHS

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
BLUE     = "#4C9BE8"
ORANGE   = "#E8834C"
GREEN    = "#4CE87A"
PURPLE   = "#B04CE8"
CYAN     = "#4CE8D8"
GRID_COL = "#2a2a3a"
TEXT_COL = "#e0e0e0"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    DARK_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "axes.titlecolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "legend.facecolor":  "#1a1a2e",
    "legend.edgecolor":  GRID_COL,
    "legend.labelcolor": TEXT_COL,
    "text.color":        TEXT_COL,
    "grid.color":        GRID_COL,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
})

def save(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {path}")

def load_wgan():
    with open("results_wgan.json") as f:
        return json.load(f)

def load_bce():
    with open("results.json") as f:
        return json.load(f)


# ── Figure W1: WGAN-GP Training Loss Curves (4 features) ─────────────────────
def fig_wgan_loss(results):
    r      = results[-1]
    nf     = r["n_features"]
    q_g    = r["qgan"]["history"]["gen_loss"]
    q_c    = r["qgan"]["history"]["disc_loss"]
    c_g    = r["classical"]["history"]["gen_loss"]
    c_c    = r["classical"]["history"]["disc_loss"]
    epochs = list(range(1, len(q_g) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"WGAN-GP Training Dynamics — HybridQGAN vs Classical GAN "
                 f"({nf} features)", fontsize=13, fontweight="bold", y=1.02)

    for ax, q, c, title in zip(
        axes,
        [q_g, q_c],
        [c_g, c_c],
        ["Generator Loss (Wasserstein)", "Critic Loss (Wasserstein)"]
    ):
        ax.plot(epochs, q, color=BLUE,   lw=2,        label="HybridQGAN")
        ax.plot(epochs, c, color=ORANGE, lw=2, ls="--", label="Classical GAN")
        ax.axhline(y=0, color=TEXT_COL, lw=0.8, ls=":", alpha=0.5)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Wasserstein Score")
        ax.grid(True)
        ax.legend()

    axes[1].annotate("Negative critic loss =\ncritic converged\n(healthy WGAN-GP)",
                     xy=(epochs[-1], q_c[-1]),
                     xytext=(epochs[-1] * 0.6, q_c[-1] + 2),
                     color=BLUE, fontsize=9,
                     arrowprops=dict(arrowstyle="->", color=BLUE))

    plt.tight_layout()
    save("fig_wgan_loss.png")


# ── Figure W2: WGAN-GP MAE Convergence ───────────────────────────────────────
def fig_wgan_mae(results):
    r      = results[-1]
    nf     = r["n_features"]
    epochs = r["qgan"]["history"]["mae_epochs"]
    q_m    = r["qgan"]["history"]["mean_MAE"]
    q_s    = r["qgan"]["history"]["std_MAE"]
    c_m    = r["classical"]["history"]["mean_MAE"]
    c_s    = r["classical"]["history"]["std_MAE"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"WGAN-GP Distribution Matching — Mean & Std MAE ({nf} features)",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, qv, cv, title in zip(
        axes,
        [q_m, q_s],
        [c_m, c_s],
        ["Mean MAE (↓ better)", "Std MAE (↓ better — quantum advantage)"]
    ):
        ax.plot(epochs, qv, color=BLUE,   lw=2, marker="o", label="HybridQGAN WGAN-GP")
        ax.plot(epochs, cv, color=ORANGE, lw=2, marker="s", ls="--", label="Classical WGAN-GP")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    save("fig_wgan_mae.png")


# ── Figure W3: WGAN-GP Classification Metrics ────────────────────────────────
def fig_wgan_metrics(results):
    metrics   = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1"]
    n_list    = [r["n_features"] for r in results]
    n_configs = len(n_list)

    fig, axes = plt.subplots(1, n_configs, figsize=(6 * n_configs, 6), sharey=True)
    if n_configs == 1:
        axes = [axes]

    fig.suptitle("WGAN-GP Classification Metrics by Feature Count",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(metrics))
    w = 0.35

    for ax, r in zip(axes, results):
        nf     = r["n_features"]
        q_vals = [r["qgan"]["clf"][m]      for m in metrics]
        c_vals = [r["classical"]["clf"][m] for m in metrics]

        b1 = ax.bar(x - w/2, q_vals, w, label="HybridQGAN WGAN-GP", color=BLUE,   alpha=0.9)
        b2 = ax.bar(x + w/2, c_vals, w, label="Classical WGAN-GP",  color=ORANGE, alpha=0.9)

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{nf} Features", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=20, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.grid(True, axis="y")
        ax.legend(fontsize=9)

    plt.tight_layout()
    save("fig_wgan_metrics.png")


# ── Figure W4: WGAN-GP Feature Sweep ─────────────────────────────────────────
def fig_wgan_sweep(results):
    n_list = [r["n_features"]                          for r in results]
    q_std  = [r["qgan"]["history"]["std_MAE"][-1]      for r in results]
    c_std  = [r["classical"]["history"]["std_MAE"][-1]  for r in results]
    q_spec = [r["qgan"]["clf"]["Specificity"]           for r in results]
    c_spec = [r["classical"]["clf"]["Specificity"]      for r in results]
    q_acc  = [r["qgan"]["clf"]["Accuracy"]              for r in results]
    c_acc  = [r["classical"]["clf"]["Accuracy"]         for r in results]
    q_f1   = [r["qgan"]["clf"]["F1"]                    for r in results]
    c_f1   = [r["classical"]["clf"]["F1"]               for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("WGAN-GP Feature Sweep — HybridQGAN vs Classical GAN",
                 fontsize=13, fontweight="bold")

    plots = [
        (axes[0, 0], q_std,  c_std,  "Std MAE (↓ better — quantum variance advantage)", True),
        (axes[0, 1], q_spec, c_spec, "Specificity (↑ better)",  False),
        (axes[1, 0], q_acc,  c_acc,  "Accuracy (↑ better)",     False),
        (axes[1, 1], q_f1,   c_f1,   "F1 Score (↑ better)",     False),
    ]

    for ax, qv, cv, title, lower in plots:
        ax.plot(n_list, qv, color=BLUE,   lw=2, marker="o", label="HybridQGAN WGAN-GP")
        ax.plot(n_list, cv, color=ORANGE, lw=2, marker="s", ls="--", label="Classical WGAN-GP")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("Value")
        ax.set_xticks(n_list)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    save("fig_wgan_sweep.png")


# ── KEY FIGURE: BCE vs WGAN-GP Comparison — the paper's main contribution ────
def fig_loss_comparison(bce_results, wgan_results):
    """
    2×2 matrix showing the key finding:
    - Quantum generator: BCE collapses, WGAN-GP fixes it
    - Classical GAN: BCE works, WGAN-GP degrades
    This is the main novel finding of the paper.
    """
    metrics = ["Accuracy", "Specificity", "F1"]
    n_list  = [r["n_features"] for r in bce_results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "KEY FINDING: Loss Function × Architecture Interaction\n"
        "Quantum generators are more compatible with WGAN-GP than classical generators",
        fontsize=13, fontweight="bold"
    )

    row_labels = ["HybridQGAN (Quantum Generator)", "Classical GAN"]
    row_keys   = ["qgan", "classical"]

    for row, (row_label, row_key) in enumerate(zip(row_labels, row_keys)):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]

            bce_vals  = [r[row_key]["clf"][metric] for r in bce_results]
            wgan_vals = [r[row_key]["clf"][metric] for r in wgan_results]

            ax.plot(n_list, bce_vals,  color=ORANGE, lw=2.5, marker="o",
                    label="BCE loss",    ms=8)
            ax.plot(n_list, wgan_vals, color=GREEN,  lw=2.5, marker="s",
                    label="WGAN-GP loss", ms=8, ls="--")

            ax.set_title(f"{row_label}\n{metric}", fontweight="bold", fontsize=10)
            ax.set_xlabel("Features")
            ax.set_ylabel(metric)
            ax.set_xticks(n_list)
            ax.set_ylim(0, 1.1)
            ax.grid(True)
            ax.legend(fontsize=9)

            # highlight direction of change
            delta = wgan_vals[-1] - bce_vals[-1]
            arrow_col = GREEN if delta > 0 else "#E84C4C"
            arrow_sym = "▲" if delta > 0 else "▼"
            ax.text(0.97, 0.08, f"{arrow_sym} {abs(delta):.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    color=arrow_col, fontsize=11, fontweight="bold")

    plt.tight_layout()
    save("fig_loss_comparison.png")


# ── Figure: Specificity Summary — all 4 model×loss combinations ──────────────
def fig_specificity_summary(bce_results, wgan_results):
    """
    Bar chart showing Specificity for all 4 combinations at 4 features.
    The most dramatic single figure for the paper.
    """
    r_bce  = bce_results[-1]   # 4 features
    r_wgan = wgan_results[-1]  # 4 features
    nf = r_bce["n_features"]

    labels = [
        "HybridQGAN\n(BCE)",
        "HybridQGAN\n(WGAN-GP)",
        "Classical GAN\n(BCE)",
        "Classical GAN\n(WGAN-GP)",
    ]
    spec_vals = [
        r_bce["qgan"]["clf"]["Specificity"],
        r_wgan["qgan"]["clf"]["Specificity"],
        r_bce["classical"]["clf"]["Specificity"],
        r_wgan["classical"]["clf"]["Specificity"],
    ]
    acc_vals = [
        r_bce["qgan"]["clf"]["Accuracy"],
        r_wgan["qgan"]["clf"]["Accuracy"],
        r_bce["classical"]["clf"]["Accuracy"],
        r_wgan["classical"]["clf"]["Accuracy"],
    ]
    colors = [BLUE, CYAN, ORANGE, PURPLE]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"4-Model Comparison at {nf} Features — Specificity & Accuracy",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, vals, title in zip(axes, [spec_vals, acc_vals], ["Specificity", "Accuracy"]):
        bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor=GRID_COL, alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontweight="bold", fontsize=11)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.15)
        ax.grid(True, axis="y")
        # highlight best
        best_idx = vals.index(max(vals))
        axes_patch = bars[best_idx]
        ax.text(axes_patch.get_x() + axes_patch.get_width()/2,
                axes_patch.get_height() + 0.06,
                "★ BEST", ha="center", color=colors[best_idx],
                fontweight="bold", fontsize=10)

    plt.tight_layout()
    save("fig_4model_summary.png")


def main():
    print("\n  Generating WGAN-GP figures from results_wgan.json...")

    wgan = load_wgan()
    fig_wgan_loss(wgan)
    fig_wgan_mae(wgan)
    fig_wgan_metrics(wgan)
    fig_wgan_sweep(wgan)

    print("\n  Generating comparison figures (BCE vs WGAN-GP)...")
    try:
        bce = load_bce()
        fig_loss_comparison(bce, wgan)
        fig_specificity_summary(bce, wgan)
        print("  Comparison figures generated successfully.")
    except FileNotFoundError:
        print("  WARNING: results.json not found — skipping comparison figures.")
        print("           Run python -m qgan.train first to generate BCE results.")

    print(f"\n  All WGAN-GP figures saved to ./{FIGURES_DIR}/")
    print("  Key figures:")
    print("    fig_wgan_loss.png        — WGAN-GP training dynamics")
    print("    fig_wgan_mae.png         — WGAN-GP MAE convergence")
    print("    fig_wgan_metrics.png     — WGAN-GP classification metrics")
    print("    fig_wgan_sweep.png       — WGAN-GP feature sweep")
    print("    fig_loss_comparison.png  — KEY: BCE vs WGAN-GP for each architecture")
    print("    fig_4model_summary.png   — KEY: all 4 models at 4 features")


if __name__ == "__main__":
    main()