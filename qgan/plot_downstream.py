# qgan/plot_downstream.py
# Generates all figures and LaTeX table for the downstream SVM evaluation
# on ANPHY-Sleep (EPCTL01, EPCTL02, EPCTL03).
#
# Run: python -m qgan.plot_downstream
#      (or:  python plot_downstream.py  from the project root)
#
# Reads:  downstream_results.json   (the JSON you already have)
# Saves:  figures/downstream_*.png  +  downstream_table.txt (corrected LaTeX)

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ── style (matches plot_noise.py exactly) ───────────────────────────────────
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

# ── colour palette ───────────────────────────────────────────────────────────
GREY   = "#9e9e9e"   # Real Only Baseline
BLUE   = "#4fc3f7"   # Classical GAN (WGAN-GP)
GREEN  = "#69f0ae"   # QGAN Noiseless
ORANGE = "#ffa726"   # QGAN Noisy

MODELS  = ["baseline", "classical_gan_aug", "qgan_noiseless_aug", "qgan_noisy_aug"]
LABELS  = ["Real Only\n(Baseline)", "Classical GAN\n(WGAN-GP)", "QGAN\nNoiseless", "QGAN\nNoisy"]
COLORS  = [GREY, BLUE, GREEN, ORANGE]
FEATS   = [2, 3, 4]

# Sleep stage label order used throughout
STAGES  = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_COLORS = ["#ef5350", "#ab47bc", "#42a5f5", "#26c6da", "#ffca28"]

os.makedirs("figures", exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────────
with open("downstream_augmentation_results.json") as f:
    data = json.load(f)

def get(n_f, model, key):
    return data[f"{n_f}_features"][model][key]

def get_f1(n_f, model, stage):
    return data[f"{n_f}_features"][model]["f1_per_class"][stage]


# ============================================================================
#  FIGURE 1 — Accuracy & Macro F1 sweep (main overview)
# ============================================================================
def fig_acc_f1_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        "Downstream SVM Evaluation — ANPHY-Sleep (EPCTL01–03)\n"
        "Accuracy & Macro F1 | All Models | 2–4 EEG Features",
        color="white", fontsize=14, fontweight="bold", y=1.02
    )

    x = np.arange(len(FEATS))
    w = 0.20
    metrics = [("Accuracy", "accuracy"), ("Macro F1", "macro_f1")]

    for ax, (title, key) in zip(axes, metrics):
        for i, (model, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
            vals = [get(n_f, model, key) for n_f in FEATS]
            bars = ax.bar(x + i * w, vals, w, label=label, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.007, f"{v:.4f}",
                        ha="center", va="bottom", fontsize=7, color="white",
                        rotation=90)

        ax.set_title(f"{title} (↑ better)", color="white")
        ax.set_xticks(x + 1.5 * w)
        ax.set_xticklabels([f"{n} Features" for n in FEATS])
        ax.set_ylim(0, 0.70)
        ax.grid(axis="y")
        ax.set_facecolor("#0d0d1a")
        if ax is axes[0]:
            ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white",
                      loc="upper left")

    plt.tight_layout()
    path = "figures/downstream_acc_f1_sweep.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
#  FIGURE 2 — Delta (improvement over baseline) bar chart
# ============================================================================
def fig_delta():
    # Only the three augmentation models have deltas
    aug_models  = MODELS[1:]
    aug_labels  = ["Classical GAN\n(WGAN-GP)", "QGAN\nNoiseless", "QGAN\nNoisy"]
    aug_colors  = COLORS[1:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        "Δ Change Over Real-Only Baseline — ANPHY-Sleep (EPCTL01–03)\n"
        "Positive = improvement | Negative = degradation",
        color="white", fontsize=13, fontweight="bold"
    )

    delta_keys = [("ΔAccuracy", "acc_improvement"), ("ΔMacro F1", "f1_improvement")]
    x = np.arange(len(FEATS))
    w = 0.25

    for ax, (title, key) in zip(axes, delta_keys):
        for i, (model, label, color) in enumerate(zip(aug_models, aug_labels, aug_colors)):
            vals = [get(n_f, model, key) for n_f in FEATS]
            bars = ax.bar(x + i * w, vals, w, label=label, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ypos = bar.get_height() + 0.001 if v >= 0 else bar.get_height() - 0.008
                ax.text(bar.get_x() + bar.get_width() / 2,
                        ypos, f"{v:+.4f}",
                        ha="center", va="bottom", fontsize=8, color="white")

        ax.axhline(0, color="#ff5555", linewidth=1.2, linestyle="--", alpha=0.8)
        ax.set_title(f"{title}", color="white")
        ax.set_xticks(x + w)
        ax.set_xticklabels([f"{n} Features" for n in FEATS])
        ax.grid(axis="y")
        ax.set_facecolor("#0d0d1a")
        ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    path = "figures/downstream_delta.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
#  FIGURE 3 — Per-class F1 heatmap grid (3 feature counts × 4 models)
# ============================================================================
def fig_per_class_f1_heatmap():
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        "Per-Class F1 Score — Downstream SVM on ANPHY-Sleep\n"
        "Rows = Feature count | Columns = Model",
        color="white", fontsize=13, fontweight="bold"
    )

    model_titles = [
        "Real Only (Baseline)",
        "Real + Classical GAN (WGAN-GP)",
        "Real + QGAN Noiseless",
        "Real + QGAN Noisy",
    ]

    for row, n_f in enumerate(FEATS):
        for col, (model, color) in enumerate(zip(MODELS, COLORS)):
            ax = axes[row][col]
            vals = [get_f1(n_f, model, s) for s in STAGES]
            bars = ax.bar(STAGES, vals, color=STAGE_COLORS, alpha=0.85, width=0.6)

            # Highlight N1 — the critical minority class
            for i, (bar, v) in enumerate(zip(bars, vals)):
                label = f"{v:.4f}"
                is_n1 = STAGES[i] == "N1"
                if is_n1:
                    bar.set_edgecolor("gold")
                    bar.set_linewidth(2.5)
                    label += "\n★N1"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, label,
                        ha="center", va="bottom", fontsize=8,
                        color="gold" if is_n1 else "white")

            ax.set_ylim(0, 0.85)
            ax.set_facecolor("#0d0d1a")
            ax.grid(axis="y")
            ax.tick_params(axis="x", labelsize=8)

            if row == 0:
                ax.set_title(model_titles[col], color=color,
                             fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{n_f} Features", color="white", fontsize=10)

    plt.tight_layout()
    path = "figures/downstream_per_class_f1.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
#  FIGURE 4 — N1 F1 focus (the key quantum advantage finding)
# ============================================================================
def fig_n1_collapse():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        "N1 (Minority Class) F1 Score — Downstream SVM on ANPHY-Sleep\n"
        "Classical GAN collapses N1 F1 → 0.000 | QGANs preserve nonzero N1 F1",
        color="white", fontsize=13, fontweight="bold"
    )

    for ax, n_f in zip(axes, FEATS):
        vals = [get_f1(n_f, model, "N1") for model in MODELS]
        short_labels = ["Baseline", "Classical\nGAN", "QGAN\nNoiseless", "QGAN\nNoisy"]
        bars = ax.bar(short_labels, vals, color=COLORS, alpha=0.85, width=0.55)

        for i, (bar, v) in enumerate(zip(bars, vals)):
            label = f"{v:.4f}"
            is_zero = v == 0.0
            if is_zero:
                label += "\n✗ COLLAPSE"
                bar.set_edgecolor("#ff5555")
                bar.set_linewidth(2.5)
            elif i > 0:  # augmented models
                label += "\n✓"
                bar.set_edgecolor("gold")
                bar.set_linewidth(1.8)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003, label,
                    ha="center", va="bottom", fontsize=9,
                    color="#ff5555" if is_zero else ("gold" if i > 0 else "white"))

        ax.set_title(f"{n_f} Features", color="white")
        ax.set_ylim(0, max(max(vals) * 1.5, 0.15))
        ax.grid(axis="y")
        ax.set_facecolor("#0d0d1a")

    plt.tight_layout()
    path = "figures/downstream_n1_collapse.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
#  FIGURE 5 — Summary: all four metrics in one panel
# ============================================================================
def fig_summary():
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        "Downstream Evaluation Summary — ANPHY-Sleep (EPCTL01–03)\n"
        "RBF-SVM | WGAN-GP for all generative models | Real PSG labels",
        color="white", fontsize=13, fontweight="bold"
    )

    panels = [
        ("Accuracy ↑",   "accuracy"),
        ("Macro F1 ↑",   "macro_f1"),
        ("N1 F1 ↑",      None),   # special — from f1_per_class
        ("ΔAcc (aug only) ↑", "acc_improvement"),
    ]

    x = np.arange(len(FEATS))
    w = 0.20

    for ax, (title, key) in zip(axes, panels):
        for i, (model, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
            if key == "acc_improvement" and model == "baseline":
                continue  # baseline has no delta
            if key is None:
                vals = [get_f1(n_f, model, "N1") for n_f in FEATS]
            else:
                try:
                    vals = [get(n_f, model, key) for n_f in FEATS]
                except KeyError:
                    continue

            bars = ax.bar(x + i * w, vals, w, label=label, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7, color="white")

        ax.set_title(title, color="white", fontweight="bold")
        ax.set_xticks(x + 1.5 * w)
        ax.set_xticklabels(["2f", "3f", "4f"])
        ax.grid(axis="y")
        ax.set_facecolor("#0d0d1a")
        if ax is axes[0]:
            ax.legend(fontsize=7.5, facecolor="#1a1a2e", labelcolor="white")
        if key == "acc_improvement":
            ax.axhline(0, color="#ff5555", linewidth=1, linestyle="--", alpha=0.7)

    plt.tight_layout()
    path = "figures/downstream_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
#  CORRECTED LaTeX TABLE
# ============================================================================
def latex_table():
    """
    Corrections applied vs. the original LaTeX you provided:
      1. Caption: "minority class (N1)" — N1 is minority (8.2%), NOT Wake (10.9%)
      2. Features=4 baseline MacroF1 0.4343 is the best → bold it
      3. All bolding re-derived from actual JSON values (no manual errors)
    """

    lines = []
    lines.append("% ============================================================")
    lines.append("% Downstream SVM Evaluation — ANPHY-Sleep (EPCTL01-03)")
    lines.append("% All three models trained with WGAN-GP for fair comparison")
    lines.append("% Real PSG labels: W=Wake, N1, N2, N3, R=REM")
    lines.append("% L (Lights) epochs excluded")
    lines.append("% Classifier: RBF-SVM, balanced class weights")
    lines.append("%")
    lines.append("% CORRECTIONS vs. original draft:")
    lines.append("%   1. Caption: minority class is N1 (8.2%), not Wake (10.9%)")
    lines.append("%   2. Features=4 baseline MacroF1 0.4343 is best → now bolded")
    lines.append("% ============================================================\n")

    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Downstream sleep stage classification on ANPHY-Sleep "
        r"(EPCTL01--03, 3 subjects) using real PSG stage labels "
        r"(W, N1, N2, N3, REM). All three generative models trained with "
        r"WGAN-GP for architectural fairness. Each augments the minority "
        r"class (N1) with synthetic samples. "  # ← CORRECTION 1
        r"Classifier: RBF-SVM with balanced class weights. "
        r"Bold = best per feature group. "
        r"$\Delta$ = change over real-data-only baseline.}"
    )
    lines.append(r"\label{tab:downstream}")
    lines.append(r"\begin{tabular}{|c|l|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Feat.} & \textbf{Training Data} & "
        r"\textbf{Acc} & $\boldsymbol{\Delta}$\textbf{Acc} & "
        r"\textbf{Macro F1} & $\boldsymbol{\Delta}$\textbf{F1} \\"
    )
    lines.append(r"\hline")

    row_labels = {
        "baseline":           "Real Only (Baseline)",
        "classical_gan_aug":  "Real + Classical GAN (WGAN-GP)",
        "qgan_noiseless_aug": "Real + QGAN Noiseless",
        "qgan_noisy_aug":     "Real + QGAN Noisy",
    }

    for n_f in FEATS:
        # find best Accuracy and best MacroF1 in this feature group
        best_acc  = max(get(n_f, m, "accuracy")  for m in MODELS)
        best_f1   = max(get(n_f, m, "macro_f1")  for m in MODELS)

        for j, model in enumerate(MODELS):
            acc  = get(n_f, model, "accuracy")
            mf1  = get(n_f, model, "macro_f1")

            acc_str = f"\\textbf{{{acc:.4f}}}" if acc == best_acc else f"{acc:.4f}"
            f1_str  = f"\\textbf{{{mf1:.4f}}}" if mf1 == best_f1  else f"{mf1:.4f}"

            if model == "baseline":
                delta_acc = "---"
                delta_f1  = "---"
            else:
                da = get(n_f, model, "acc_improvement")
                df = get(n_f, model, "f1_improvement")
                delta_acc = f"{da:+.4f}"
                delta_f1  = f"{df:+.4f}"

            lines.append(
                f"  {row_labels[model]} & "
                f"{acc_str} & {delta_acc} & "
                f"{f1_str} & {delta_f1} \\\\"
            )

        lines.append(r"  \hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = "downstream_table.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out}")

    # ── print correction summary ─────────────────────────────────────────────
    print("\n  CORRECTIONS APPLIED TO LaTeX TABLE:")
    print("  ─────────────────────────────────────────────────────────")
    print("  1. Caption: 'minority class (N1)' — N1 is 8.2% of data,")
    print("     Wake is 10.9%. Original draft said Wake which was wrong.")
    print("  2. Features=4: Baseline MacroF1 = 0.4343 is the highest")
    print("     in that group — now correctly bolded.")
    print("  ─────────────────────────────────────────────────────────\n")


# ============================================================================
#  MAIN
# ============================================================================
def main():
    print("\n  Generating downstream evaluation figures and table...")
    print(f"  {'='*60}")

    fig_acc_f1_sweep()
    fig_delta()
    fig_per_class_f1_heatmap()
    fig_n1_collapse()
    fig_summary()
    latex_table()

    print(f"  {'='*60}")
    print("\n  All figures saved to figures/")
    print("  Corrected LaTeX table saved to downstream_table.txt\n")

    # ── quick results summary ────────────────────────────────────────────────
    print("  QUICK RESULTS SUMMARY")
    print(f"  {'─'*70}")
    print(f"  {'Feat':<6} {'Model':<32} {'Acc':<8} {'MacroF1':<10} {'N1 F1'}")
    print(f"  {'─'*70}")
    short = {
        "baseline":           "Real Only (Baseline)          ",
        "classical_gan_aug":  "Real + Classical GAN (WGAN-GP)",
        "qgan_noiseless_aug": "Real + QGAN Noiseless         ",
        "qgan_noisy_aug":     "Real + QGAN Noisy             ",
    }
    for n_f in FEATS:
        for model in MODELS:
            acc = get(n_f, model, "accuracy")
            mf1 = get(n_f, model, "macro_f1")
            n1  = get_f1(n_f, model, "N1")
            n1_flag = "  ← COLLAPSE" if n1 == 0.0 else ""
            print(f"  {str(n_f)+'f':<6} {short[model]} "
                  f"{acc:<8.4f} {mf1:<10.4f} {n1:.4f}{n1_flag}")
        print(f"  {'─'*70}")


if __name__ == "__main__":
    main()