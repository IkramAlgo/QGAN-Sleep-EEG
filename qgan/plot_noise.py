# qgan/plot_noise.py
# Generates all figures and LaTeX tables for the noise experiment
# Run: python -m qgan.plot_noise
#
# Reads: results_noise.json
# Saves: figures/noise_*.png  +  results_noise_table.txt (LaTeX)

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d0d1a",
    "axes.facecolor":    "#0d0d1a",
    "axes.edgecolor":    "#444466",
    "axes.labelcolor":   "#ccccdd",
    "xtick.color":       "#ccccdd",
    "ytick.color":       "#ccccdd",
    "text.color":        "#ccccdd",
    "grid.color":        "#222244",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
})

BLUE   = "#4fc3f7"   # Classical GAN
GREEN  = "#69f0ae"   # QGAN Noiseless
ORANGE = "#ffa726"   # QGAN Noisy
MODELS = ["classical_gan", "qgan_noiseless", "qgan_noisy"]
LABELS = ["Classical GAN", "QGAN Noiseless", "QGAN Noisy"]
COLORS = [BLUE, GREEN, ORANGE]
FEATS  = [2, 3, 4]

os.makedirs("figures", exist_ok=True)

# ── load data ────────────────────────────────────────────────────
with open("results_noise.json") as f:
    data = json.load(f)

def get(n_f, model, key1, key2=None):
    d = data[f"{n_f}_features"][model]
    return d[key1] if key2 is None else d[key1][key2]


# ================================================================
#  FIGURE 1 — Feature Sweep: 5 metrics × 3 models × 3 feature counts
# ================================================================
def fig_feature_sweep():
    metrics = [
        ("Accuracy",    "clf", "Accuracy",    "↑ better", 0, 1),
        ("Specificity", "clf", "Specificity",  "↑ better", 0, 1),
        ("F1 Score",    "clf", "F1",           "↑ better", 0, 1),
        ("Mean MAE",    "mae", "mean_MAE",     "↓ better", 0, None),
        ("Std MAE",     "mae", "std_MAE",      "↓ better", 0, None),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        "Noise Experiment — Feature Sweep (2, 3, 4 Features)\n"
        "Classical GAN vs QGAN Noiseless vs QGAN Noisy Data",
        color="white", fontsize=14, fontweight="bold", y=1.02
    )

    x = np.arange(len(FEATS))
    w = 0.25

    for ax, (title, src, key, direction, ymin, ymax) in zip(axes, metrics):
        for i, (model, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
            vals = [get(n_f, model, src, key) for n_f in FEATS]
            bars = ax.bar(x + i*w, vals, w, label=label, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.5, color="white")

        ax.set_title(f"{title}\n({direction})", color="white")
        ax.set_xticks(x + w)
        ax.set_xticklabels([f"{n}f" for n in FEATS])
        ax.set_ylim(ymin, (ymax or max(
            get(n_f, m, src, key) for n_f in FEATS for m in MODELS
        ) * 1.25))
        ax.grid(axis="y")
        ax.set_facecolor("#0d0d1a")
        if ax is axes[0]:
            ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    path = "figures/noise_feature_sweep.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
#  FIGURE 2 — StdMAE Comparison (Quantum Variance Advantage)
# ================================================================
def fig_stdmae():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        "Std MAE — Quantum Variance Advantage\n"
        "Lower = generator matches EEG variance distribution better",
        color="white", fontsize=13, fontweight="bold"
    )

    for ax, n_f in zip(axes, FEATS):
        vals   = [get(n_f, m, "mae", "std_MAE") for m in MODELS]
        best_i = int(np.argmin(vals))
        bars   = ax.bar(LABELS, vals, color=COLORS, alpha=0.85, width=0.5)

        for i, (bar, v) in enumerate(zip(bars, vals)):
            label = f"{v:.4f}"
            if i == best_i:
                label += "\n★ BEST"
                bar.set_edgecolor("gold")
                bar.set_linewidth(2)
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005, label,
                    ha="center", va="bottom", fontsize=9,
                    color="gold" if i == best_i else "white")

        ax.set_title(f"{n_f} Features", color="white")
        ax.set_ylim(0, max(vals) * 1.3)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Classical\nGAN", "QGAN\nNoiseless", "QGAN\nNoisy"],
                           fontsize=9)
        ax.grid(axis="y")
        ax.set_facecolor("#0d0d1a")

    plt.tight_layout()
    path = "figures/noise_stdmae.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
#  FIGURE 3 — Classification Metrics Summary (3-panel per feature)
# ================================================================
def fig_classification():
    clf_keys = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1"]
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        "Classification Metrics — Noise Experiment (50 Epochs)\n"
        "Rows = Feature counts | Columns = Metrics",
        color="white", fontsize=13, fontweight="bold"
    )

    x = np.arange(3)
    for row, n_f in enumerate(FEATS):
        for col, metric in enumerate(clf_keys):
            ax   = axes[row][col]
            vals = [get(n_f, m, "clf", metric) for m in MODELS]
            bars = ax.bar(x, vals, color=COLORS, alpha=0.85, width=0.6)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8, color="white")
            ax.set_ylim(0, 1.25)
            ax.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(["Classical", "Noiseless", "Noisy"], fontsize=7.5)
            ax.set_facecolor("#0d0d1a")
            ax.grid(axis="y")
            if row == 0:
                ax.set_title(metric, color="white", fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{n_f} features", color="white", fontsize=10)

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for c, l in zip(COLORS, LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               facecolor="#1a1a2e", labelcolor="white", fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = "figures/noise_classification.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
#  FIGURE 4 — Training Loss Curves (StdMAE over epochs)
# ================================================================
def fig_mae_curves():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        "MAE Convergence over 50 Epochs — Noise Experiment",
        color="white", fontsize=13, fontweight="bold"
    )

    for col, n_f in enumerate(FEATS):
        # Mean MAE
        ax_mean = axes[0][col]
        for model, label, color in zip(MODELS, LABELS, COLORS):
            epochs = data[f"{n_f}_features"][model]["history"]["mae_epochs"]
            vals   = data[f"{n_f}_features"][model]["history"]["mean_MAE"]
            ax_mean.plot(epochs, vals, color=color, label=label,
                         linewidth=1.8, alpha=0.85)
        ax_mean.set_title(f"{n_f} Features — Mean MAE", color="white")
        ax_mean.set_xlabel("Epoch")
        ax_mean.set_ylabel("Mean MAE ↓")
        ax_mean.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
        ax_mean.grid()
        ax_mean.set_facecolor("#0d0d1a")

        # Std MAE
        ax_std = axes[1][col]
        for model, label, color in zip(MODELS, LABELS, COLORS):
            epochs = data[f"{n_f}_features"][model]["history"]["mae_epochs"]
            vals   = data[f"{n_f}_features"][model]["history"]["std_MAE"]
            ax_std.plot(epochs, vals, color=color, label=label,
                        linewidth=1.8, alpha=0.85)
        ax_std.set_title(f"{n_f} Features — Std MAE", color="white")
        ax_std.set_xlabel("Epoch")
        ax_std.set_ylabel("Std MAE ↓")
        ax_std.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
        ax_std.grid()
        ax_std.set_facecolor("#0d0d1a")

    plt.tight_layout()
    path = "figures/noise_mae_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
#  FIGURE 5 — Summary: Best model per metric per feature count
# ================================================================
def fig_summary():
    metrics = [
        ("Accuracy ↑",    "clf", "Accuracy",    True),
        ("Specificity ↑", "clf", "Specificity",  True),
        ("F1 Score ↑",    "clf", "F1",           True),
        ("Std MAE ↓",     "mae", "std_MAE",      False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        "Noise Experiment — Summary (Best Result per Metric)\n"
        "Arch C + WGAN-GP | Gaussian Noise Level = 0.1 | 50 Epochs",
        color="white", fontsize=13, fontweight="bold"
    )

    x     = np.arange(len(FEATS))
    w     = 0.25

    for ax, (title, src, key, higher_better) in zip(axes, metrics):
        for i, (model, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
            vals = [get(n_f, model, src, key) for n_f in FEATS]
            bars = ax.bar(x + i*w, vals, w, label=label, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.5, color="white")

        ax.set_title(title, color="white", fontweight="bold")
        ax.set_xticks(x + w)
        ax.set_xticklabels(["2f", "3f", "4f"])
        ax.grid(axis="y")
        ax.set_facecolor("#0d0d1a")
        ax.legend(fontsize=7.5, facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    path = "figures/noise_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
#  LATEX TABLES
# ================================================================
def latex_tables():
    lines = []
    lines.append("% ============================================================")
    lines.append("% NOISE EXPERIMENT RESULTS — LaTeX Tables")
    lines.append("% Classical GAN vs QGAN Noiseless vs QGAN Noisy (Arch C WGAN-GP)")
    lines.append("% Gaussian noise level = 0.1 | 50 epochs | features 2, 3, 4")
    lines.append("% ============================================================\n")

    for n_f in FEATS:
        feat_names = ", ".join(data[f"{n_f}_features"]["feature_names"])
        lines.append(f"% ── {n_f} FEATURES ({feat_names}) ──\n")
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(
            f"\\caption{{Noise Experiment — {n_f} Features ({feat_names}). "
            "Gaussian noise $\\sigma=0.1$ applied to training data. "
            "QGAN Noisy evaluated on clean data.}}"
        )
        lines.append(r"\label{tab:noise_" + str(n_f) + r"f}")
        lines.append(r"\begin{tabular}{|l|c|c|c|c|c|c|c|}")
        lines.append(r"\hline")
        lines.append(
            r"\textbf{Model} & \textbf{Acc} & \textbf{Prec} & "
            r"\textbf{Sens} & \textbf{Spec} & \textbf{F1} & "
            r"\textbf{MeanMAE} & \textbf{StdMAE} \\"
        )
        lines.append(r"\hline")

        model_display = {
            "classical_gan":  "Classical GAN",
            "qgan_noiseless": "QGAN Noiseless",
            "qgan_noisy":     "QGAN Noisy",
        }

        # find best StdMAE for bold
        std_maes = [get(n_f, m, "mae", "std_MAE") for m in MODELS]
        best_std = min(std_maes)
        best_acc = max(get(n_f, m, "clf", "Accuracy") for m in MODELS)

        for model in MODELS:
            clf = data[f"{n_f}_features"][model]["clf"]
            mae = data[f"{n_f}_features"][model]["mae"]
            acc = clf["Accuracy"]
            std = mae["std_MAE"]

            acc_str = f"\\textbf{{{acc:.4f}}}" if acc == best_acc else f"{acc:.4f}"
            std_str = f"\\textbf{{{std:.4f}}}" if std == best_std else f"{std:.4f}"

            lines.append(
                f"  {model_display[model]} & "
                f"{acc_str} & "
                f"{clf['Precision']:.4f} & "
                f"{clf['Sensitivity']:.4f} & "
                f"{clf['Specificity']:.4f} & "
                f"{clf['F1']:.4f} & "
                f"{mae['mean_MAE']:.4f} & "
                f"{std_str} \\\\"
            )
            lines.append(r"  \hline")

        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}" + "\n")

    # Combined summary table across all feature counts
    lines.append("% ── COMBINED SUMMARY TABLE ──\n")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Noise Experiment Summary — All Feature Counts. "
        r"QGAN uses Arch C (6-qubit ring CNOT, RX$\rightarrow$CNOT$\rightarrow$RY, WGAN-GP). "
        r"Gaussian noise $\sigma=0.1$.}"
    )
    lines.append(r"\label{tab:noise_summary}")
    lines.append(r"\begin{tabular}{|c|l|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Features} & \textbf{Model} & \textbf{Accuracy} & "
        r"\textbf{Specificity} & \textbf{F1} & \textbf{MeanMAE} & \textbf{StdMAE} \\"
    )
    lines.append(r"\hline")

    for n_f in FEATS:
        for j, model in enumerate(MODELS):
            clf = data[f"{n_f}_features"][model]["clf"]
            mae = data[f"{n_f}_features"][model]["mae"]
            feat_cell = f"\\multirow{{3}}{{*}}{{{n_f}}}" if j == 0 else ""
            lines.append(
                f"  {feat_cell} & {model_display[model]} & "
                f"{clf['Accuracy']:.4f} & "
                f"{clf['Specificity']:.4f} & "
                f"{clf['F1']:.4f} & "
                f"{mae['mean_MAE']:.4f} & "
                f"{mae['std_MAE']:.4f} \\\\"
            )
        lines.append(r"  \hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = "results_noise_table.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out}")


# ================================================================
#  MAIN
# ================================================================
def main():
    print("\n  Generating noise experiment figures and tables...")
    print(f"  {'='*55}")

    fig_feature_sweep()
    fig_stdmae()
    fig_classification()
    fig_mae_curves()
    fig_summary()
    latex_tables()

    print(f"  {'='*55}")
    print("\n  All figures saved to figures/")
    print("  LaTeX tables saved to results_noise_table.txt\n")

    # Print quick summary
    print("  QUICK RESULTS SUMMARY")
    print(f"  {'─'*60}")
    print(f"  {'Features':<10} {'Model':<18} {'Acc':<7} {'Spec':<7} "
          f"{'F1':<7} {'StdMAE'}")
    print(f"  {'─'*60}")
    model_display = {
        "classical_gan":  "Classical GAN ",
        "qgan_noiseless": "QGAN Noiseless",
        "qgan_noisy":     "QGAN Noisy    ",
    }
    for n_f in FEATS:
        for model in MODELS:
            clf = data[f"{n_f}_features"][model]["clf"]
            mae = data[f"{n_f}_features"][model]["mae"]
            print(f"  {str(n_f)+'f':<10} {model_display[model]:<18} "
                  f"{clf['Accuracy']:<7} {clf['Specificity']:<7} "
                  f"{clf['F1']:<7} {mae['std_MAE']}")
        print(f"  {'─'*60}")


if __name__ == "__main__":
    main()