# qgan/plot.py
# Generates all paper figures from results.json (BCE HybridQGAN experiment)
# Run AFTER training: python -m qgan.plot
# FIX: removed qpu_avg_time key (no longer in results.json)

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from qgan.config import FIGURES_DIR, EPOCHS, EVAL_EVERY

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
BLUE     = "#4C9BE8"
ORANGE   = "#E8834C"
GREEN    = "#4CE87A"
PURPLE   = "#B04CE8"
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

def load_results():
    with open("results.json") as f:
        return json.load(f)


# ── Figure 1: Training Loss Curves (4-feature experiment) ────────────────────
def fig_training_loss(results):
    r    = results[-1]
    nf   = r["n_features"]
    q_g  = r["qgan"]["history"]["gen_loss"]
    q_d  = r["qgan"]["history"]["disc_loss"]
    c_g  = r["classical"]["history"]["gen_loss"]
    c_d  = r["classical"]["history"]["disc_loss"]
    epochs = list(range(1, len(q_g) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Figure 1: Training Loss Curves — HybridQGAN vs Classical GAN "
                 f"({nf} features)", fontsize=13, fontweight="bold", y=1.02)

    for ax, q, c, title in zip(
        axes,
        [q_g, q_d],
        [c_g, c_d],
        ["Generator Loss", "Discriminator Loss"]
    ):
        ax.plot(epochs, q, color=BLUE,   lw=2,        label="HybridQGAN (BCE)")
        ax.plot(epochs, c, color=ORANGE, lw=2, ls="--", label="Classical GAN (BCE)")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    save("fig1_training_loss.png")


# ── Figure 2: MAE Convergence ────────────────────────────────────────────────
def fig_mae_convergence(results):
    r      = results[-1]
    nf     = r["n_features"]
    epochs = r["qgan"]["history"]["mae_epochs"]
    q_m    = r["qgan"]["history"]["mean_MAE"]
    q_s    = r["qgan"]["history"]["std_MAE"]
    c_m    = r["classical"]["history"]["mean_MAE"]
    c_s    = r["classical"]["history"]["std_MAE"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Figure 2: Distribution Matching — Mean & Std MAE over Training "
                 f"({nf} features)", fontsize=13, fontweight="bold", y=1.02)

    for ax, qv, cv, title in zip(
        axes,
        [q_m, q_s],
        [c_m, c_s],
        ["Mean MAE (↓ better — mean matching)", "Std MAE (↓ better — variance matching)"]
    ):
        ax.plot(epochs, qv, color=BLUE,   lw=2, marker="o", label="HybridQGAN")
        ax.plot(epochs, cv, color=ORANGE, lw=2, marker="s", ls="--", label="Classical GAN")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    save("fig2_mae_convergence.png")


# ── Figure 3: Feature Sweep ───────────────────────────────────────────────────
def fig_feature_sweep(results):
    n_list = [r["n_features"]                          for r in results]
    q_mean = [r["qgan"]["history"]["mean_MAE"][-1]     for r in results]
    q_std  = [r["qgan"]["history"]["std_MAE"][-1]      for r in results]
    c_mean = [r["classical"]["history"]["mean_MAE"][-1] for r in results]
    c_std  = [r["classical"]["history"]["std_MAE"][-1]  for r in results]
    q_spec = [r["qgan"]["clf"]["Specificity"]           for r in results]
    c_spec = [r["classical"]["clf"]["Specificity"]      for r in results]
    q_f1   = [r["qgan"]["clf"]["F1"]                    for r in results]
    c_f1   = [r["classical"]["clf"]["F1"]               for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Figure 3: Feature Sweep — HybridQGAN vs Classical GAN (BCE)",
                 fontsize=13, fontweight="bold")

    plots = [
        (axes[0, 0], q_mean, c_mean, "Mean MAE (↓ better)", True),
        (axes[0, 1], q_std,  c_std,  "Std MAE (↓ better)",  True),
        (axes[1, 0], q_spec, c_spec, "Specificity (↑ better)", False),
        (axes[1, 1], q_f1,   c_f1,   "F1 Score (↑ better)",    False),
    ]

    for ax, qv, cv, title, lower in plots:
        ax.plot(n_list, qv, color=BLUE,   lw=2, marker="o", label="HybridQGAN")
        ax.plot(n_list, cv, color=ORANGE, lw=2, marker="s", ls="--", label="Classical GAN")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("Value")
        ax.set_xticks(n_list)
        ax.grid(True)
        ax.legend()
        best_q     = min(qv) if lower else max(qv)
        best_c     = min(cv) if lower else max(cv)
        ax.scatter([n_list[qv.index(best_q)]], [best_q], color=BLUE,   s=150, zorder=5)
        ax.scatter([n_list[cv.index(best_c)]], [best_c], color=ORANGE, s=150, zorder=5)

    plt.tight_layout()
    save("fig3_feature_sweep.png")


# ── Figure 4: Timing (no QPU estimate — uses actual BCE training times) ───────
def fig_timing(results):
    n_list  = [r["n_features"]                        for r in results]
    q_times = [r["qgan"]["history"]["avg_time"]       for r in results]
    c_times = [r["classical"]["history"]["avg_time"]  for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 4: Training Time per Epoch — HybridQGAN vs Classical GAN",
                 fontsize=13, fontweight="bold", y=1.02)

    x = np.arange(len(n_list))
    w = 0.35

    # left: grouped bar chart
    ax = axes[0]
    b1 = ax.bar(x - w/2, q_times, w, color=BLUE,   label="HybridQGAN", alpha=0.9)
    b2 = ax.bar(x + w/2, c_times, w, color=ORANGE, label="Classical GAN", alpha=0.9)
    for bar, val in list(zip(b1, q_times)) + list(zip(b2, c_times)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}f" for n in n_list])
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Avg Time per Epoch (s)")
    ax.set_title("Absolute Epoch Time", fontweight="bold")
    ax.grid(True, axis="y")
    ax.legend()

    # right: log scale
    ax2 = axes[1]
    b3 = ax2.bar(x - w/2, q_times, w, color=BLUE,   label="HybridQGAN", alpha=0.9)
    b4 = ax2.bar(x + w/2, c_times, w, color=ORANGE, label="Classical GAN", alpha=0.9)
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n}f" for n in n_list])
    ax2.set_xlabel("Number of Features")
    ax2.set_ylabel("Avg Time (s) — log scale")
    ax2.set_title("Epoch Time — Log Scale (shows quantum overhead)", fontweight="bold")
    ax2.grid(True, axis="y")
    ax2.legend()

    overhead = round(q_times[-1] / c_times[-1])
    ax2.annotate(f"HybridQGAN\n~{overhead}× slower\nthan Classical",
                 xy=(x[-1] - w/2, q_times[-1]),
                 xytext=(x[-1] - 0.8, q_times[-1] * 0.3),
                 color=BLUE, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=BLUE))

    plt.tight_layout()
    save("fig4_timing.png")


# ── Figure 5: Classification Metrics by Feature Count ────────────────────────
def fig_classification(results):
    metrics   = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1"]
    n_list    = [r["n_features"] for r in results]
    n_configs = len(n_list)

    fig, axes = plt.subplots(1, n_configs, figsize=(6 * n_configs, 6), sharey=True)
    if n_configs == 1:
        axes = [axes]

    fig.suptitle("Figure 5: Classification Metrics by Feature Count — BCE",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(metrics))
    w = 0.35

    for ax, r in zip(axes, results):
        nf     = r["n_features"]
        q_vals = [r["qgan"]["clf"][m]      for m in metrics]
        c_vals = [r["classical"]["clf"][m] for m in metrics]

        b1 = ax.bar(x - w/2, q_vals, w, label="HybridQGAN", color=BLUE,   alpha=0.9)
        b2 = ax.bar(x + w/2, c_vals, w, label="Classical",  color=ORANGE, alpha=0.9)

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
    save("fig5_classification_metrics.png")


# ── Figure 6: Final Summary Bars ──────────────────────────────────────────────
def fig_summary_bars(results):
    r  = results[-1]
    nf = r["n_features"]

    metrics_data = [
        ("Mean MAE↓",  r["qgan"]["history"]["mean_MAE"][-1],
                       r["classical"]["history"]["mean_MAE"][-1], True),
        ("Std MAE↓",   r["qgan"]["history"]["std_MAE"][-1],
                       r["classical"]["history"]["std_MAE"][-1], True),
        ("Specificity↑", r["qgan"]["clf"]["Specificity"],
                          r["classical"]["clf"]["Specificity"], False),
        ("F1 Score↑",  r["qgan"]["clf"]["F1"],
                       r["classical"]["clf"]["F1"], False),
        ("Accuracy↑",  r["qgan"]["clf"]["Accuracy"],
                       r["classical"]["clf"]["Accuracy"], False),
    ]

    fig, axes = plt.subplots(1, len(metrics_data), figsize=(18, 5))
    fig.suptitle(f"Figure 6: Final Performance Summary — HybridQGAN vs Classical GAN "
                 f"({nf} features, BCE)", fontsize=13, fontweight="bold", y=1.02)

    for ax, (title, qv, cv, lower_better) in zip(axes, metrics_data):
        qgan_wins    = (qv < cv) if lower_better else (qv > cv)
        winner_label = "Winner: HybridQGAN" if qgan_wins else "Winner: Classical"
        winner_col   = BLUE if qgan_wins else ORANGE

        bars = ax.bar(["HybridQGAN", "Classical"], [qv, cv],
                      color=[BLUE, ORANGE], width=0.5, edgecolor=GRID_COL)
        for bar, val in zip(bars, [qv, cv]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y")
        ax.text(0.5, 0.94, winner_label, transform=ax.transAxes,
                ha="center", va="top", fontsize=9, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=winner_col, alpha=0.85))

    plt.tight_layout()
    save("fig6_summary.png")


def main():
    print("\n  Generating BCE figures from results.json...")
    results = load_results()

    fig_training_loss(results)
    fig_mae_convergence(results)
    fig_feature_sweep(results)
    fig_timing(results)
    fig_classification(results)
    fig_summary_bars(results)

    print(f"\n  All figures saved to ./{FIGURES_DIR}/")


if __name__ == "__main__":
    main()