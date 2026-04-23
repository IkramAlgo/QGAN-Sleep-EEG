# qgan/plot_downstream_v2.py
# Publication-quality figures for the downstream SVM evaluation.
# Style: white background, clean IEEE/Nature aesthetic, large fonts.
#
# Run: python -m qgan.plot_downstream_v2
#
# Reads:  downstream_augmentation_results.json
# Saves:  figures/ds_*.png  +  downstream_table_corrected.txt

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import ticker

# ── Publication style (white background, IEEE-compatible) ───────────────────
matplotlib.rcParams.update({
    "figure.facecolor":       "white",
    "axes.facecolor":         "white",
    "axes.edgecolor":         "#333333",
    "axes.labelcolor":        "#111111",
    "axes.spines.top":        False,
    "axes.spines.right":      False,
    "xtick.color":            "#333333",
    "ytick.color":            "#333333",
    "text.color":             "#111111",
    "grid.color":             "#dddddd",
    "grid.linestyle":         "-",
    "grid.alpha":             0.7,
    "font.family":            "sans-serif",
    "font.sans-serif":        ["DejaVu Sans"],
    "font.size":              12,
    "axes.titlesize":         13,
    "axes.titleweight":       "bold",
    "axes.labelsize":         12,
    "xtick.labelsize":        11,
    "ytick.labelsize":        11,
    "legend.fontsize":        10,
    "legend.framealpha":      0.92,
    "legend.edgecolor":       "#cccccc",
    "figure.dpi":             150,
    "savefig.dpi":            200,
    "savefig.bbox":           "tight",
    "savefig.facecolor":      "white",
})

# ── Colour palette — colourblind-friendly ────────────────────────────────────
C_BASE    = "#546e7a"   # Blue-grey  — Baseline
C_CGAN    = "#e65100"   # Deep orange — Classical GAN
C_QNOISE  = "#1565c0"   # Deep blue   — QGAN Noiseless
C_QNOISY  = "#2e7d32"   # Deep green  — QGAN Noisy

COLORS  = [C_BASE, C_CGAN, C_QNOISE, C_QNOISY]
MODELS  = ["baseline", "classical_gan_aug", "qgan_noiseless_aug", "qgan_noisy_aug"]
LABELS  = ["Real Only\n(Baseline)", "Classical GAN\n(WGAN-GP)", "QGAN\nNoiseless", "QGAN\nNoisy"]
LABELS_SHORT = ["Baseline", "Classical GAN", "QGAN Noiseless", "QGAN Noisy"]
FEATS   = [2, 3, 4]
STAGES  = ["Wake", "N1", "N2", "N3", "REM"]

# Stage colours — distinguishable
SC = ["#d32f2f", "#7b1fa2", "#1976d2", "#00796b", "#f57f17"]

os.makedirs("figures", exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
with open("downstream_augmentation_results.json") as f:
    DATA = json.load(f)

def get(nf, model, key):
    return DATA[f"{nf}_features"][model][key]

def get_f1(nf, model, stage):
    return DATA[f"{nf}_features"][model]["f1_per_class"][stage]


# ============================================================================
#  FIGURE 1 — Main accuracy & macro-F1 overview (grouped bar, 2×1)
# ============================================================================
def fig1_overview():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Downstream Sleep-Stage Classification — ANPHY-Sleep (EPCTL01–03)\n"
        "RBF-SVM | WGAN-GP for all generative models | Real PSG labels",
        fontsize=13, fontweight="bold", y=1.01
    )

    x  = np.arange(len(FEATS))
    w  = 0.19
    metrics = [("Accuracy", "accuracy"), ("Macro F1", "macro_f1")]

    for ax, (title, key) in zip(axes, metrics):
        for i, (model, label, color) in enumerate(zip(MODELS, LABELS_SHORT, COLORS)):
            vals = [get(nf, model, key) for nf in FEATS]
            offset = (i - 1.5) * w
            bars = ax.bar(x + offset, vals, w, label=label,
                          color=color, alpha=0.88, zorder=3,
                          edgecolor="white", linewidth=0.6)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.006,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7.5, color=color, fontweight="bold",
                        rotation=90)

        ax.set_title(f"{title}  (↑ higher is better)", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n} EEG Features" for n in FEATS])
        ax.set_ylim(0, 0.72)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", zorder=0)
        ax.set_ylabel(title)

    axes[0].legend(loc="upper left", ncol=2, handlelength=1.2)

    plt.tight_layout()
    path = "figures/ds_fig1_overview.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
#  FIGURE 2 — Delta bars (augmentation effect)
# ============================================================================
def fig2_delta():
    aug_models  = MODELS[1:]
    aug_labels  = LABELS_SHORT[1:]
    aug_colors  = COLORS[1:]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Augmentation Effect vs. Real-Only Baseline — ANPHY-Sleep\n"
        "Positive Δ = improvement | Negative Δ = degradation",
        fontsize=13, fontweight="bold", y=1.01
    )

    x = np.arange(len(FEATS))
    w = 0.24
    delta_panels = [("ΔAccuracy", "acc_improvement"), ("ΔMacro F1", "f1_improvement")]

    for ax, (title, key) in zip(axes, delta_panels):
        for i, (model, label, color) in enumerate(zip(aug_models, aug_labels, aug_colors)):
            vals = [get(nf, model, key) for nf in FEATS]
            offset = (i - 1) * w
            bars = ax.bar(x + offset, vals, w, label=label,
                          color=color, alpha=0.88, zorder=3,
                          edgecolor="white", linewidth=0.6)
            for bar, v in zip(bars, vals):
                ypos = bar.get_height() + 0.001 if v >= 0 else bar.get_height() - 0.009
                ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                        f"{v:+.3f}", ha="center", va="bottom",
                        fontsize=8.5, color=color, fontweight="bold")

        ax.axhline(0, color="#333333", linewidth=1.2, linestyle="--", zorder=4)
        ax.set_title(f"{title}", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n} Features" for n in FEATS])
        ax.grid(axis="y", zorder=0)
        ax.set_ylabel(title)
        ax.legend(loc="upper left", handlelength=1.2)

        # shade the positive region lightly
        ylim = ax.get_ylim()
        ax.axhspan(0, max(0.12, ylim[1]), alpha=0.04, color="green", zorder=0)
        ax.axhspan(min(-0.05, ylim[0]), 0, alpha=0.04, color="red",   zorder=0)

    plt.tight_layout()
    path = "figures/ds_fig2_delta.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
#  FIGURE 3 — N1 minority-class collapse (KEY FINDING — standalone)
# ============================================================================
def fig3_n1_collapse():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)
    fig.suptitle(
        "N1 Minority-Class F1 — Key Quantum Advantage Finding\n"
        "Classical GAN collapses N1 F1 → 0.000 across ALL feature counts  |  "
        "Both QGANs preserve nonzero N1 F1",
        fontsize=13, fontweight="bold", y=1.02
    )

    for ax, nf in zip(axes, FEATS):
        vals = [get_f1(nf, model, "N1") for model in MODELS]
        x_pos = np.arange(len(MODELS))

        for i, (v, color, label) in enumerate(zip(vals, COLORS, LABELS_SHORT)):
            bar = ax.bar(i, v, color=color, alpha=0.88, width=0.55,
                         zorder=3, edgecolor="white", linewidth=0.7)

            is_collapse = (v == 0.0)
            is_qgan     = (i >= 2)

            if is_collapse:
                # Red hatching for collapsed bar — draw a visible stub
                ax.bar(i, 0.008, color="#ff1744", alpha=0.9, width=0.55,
                       zorder=4, hatch="///", edgecolor="#ff1744")
                ax.text(i, 0.012, "✗ COLLAPSE\n0.000",
                        ha="center", va="bottom", fontsize=9,
                        color="#c62828", fontweight="bold")
            else:
                ax.text(i, v + 0.005, f"{v:.4f}",
                        ha="center", va="bottom", fontsize=10,
                        color=color, fontweight="bold")
                if is_qgan:
                    ax.text(i, v / 2, "✓",
                            ha="center", va="center", fontsize=14,
                            color="white", fontweight="bold")

        # Baseline N1 reference line
        baseline_n1 = get_f1(nf, "baseline", "N1")
        ax.axhline(baseline_n1, color=C_BASE, linewidth=1.4,
                   linestyle=":", alpha=0.8, zorder=5,
                   label=f"Baseline N1 F1 = {baseline_n1:.4f}")

        ax.set_title(f"{nf} EEG Features", pad=8, fontsize=13)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(LABELS_SHORT, fontsize=9.5)
        ax.set_ylim(0, max(max(vals), baseline_n1) * 1.6 + 0.04)
        ax.set_ylabel("N1 Class F1 Score")
        ax.grid(axis="y", zorder=0)
        ax.legend(loc="upper right", fontsize=9)

        # Annotate the gap
        if nf >= 3:
            best_qgan = max(get_f1(nf, m, "N1") for m in MODELS[2:])
            ax.annotate(
                f"QGAN advantage:\n+{best_qgan:.4f} vs 0.000",
                xy=(2.5, best_qgan / 2),
                fontsize=8.5, color="#1565c0",
                ha="center", style="italic"
            )

    plt.tight_layout()
    path = "figures/ds_fig3_n1_collapse.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
#  FIGURE 4 — Per-class F1 heatmap grid (3 rows × 4 cols)
# ============================================================================
def fig4_perclass_grid():
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle(
        "Per-Class F1 Score — Downstream SVM on ANPHY-Sleep\n"
        "Rows = EEG Feature Count  |  Columns = Generative Model  |  ★ = N1 minority class",
        fontsize=13, fontweight="bold", y=1.01
    )

    col_titles = [
        "Real Only (Baseline)", "Real + Classical GAN\n(WGAN-GP)",
        "Real + QGAN Noiseless", "Real + QGAN Noisy",
    ]

    for row, nf in enumerate(FEATS):
        for col, (model, model_color) in enumerate(zip(MODELS, COLORS)):
            ax = axes[row][col]
            vals  = [get_f1(nf, model, s) for s in STAGES]
            x_pos = np.arange(len(STAGES))

            bars = ax.bar(x_pos, vals, color=SC, alpha=0.85,
                          width=0.62, zorder=3, edgecolor="white", linewidth=0.5)

            for i, (bar, v, stage) in enumerate(zip(bars, vals, STAGES)):
                is_n1 = (stage == "N1")
                if is_n1:
                    bar.set_edgecolor("#f57f17")
                    bar.set_linewidth(2.5)
                ypos = v + 0.015
                label = f"{v:.3f}"
                if is_n1:
                    label = f"★ {v:.3f}"
                ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                        label, ha="center", va="bottom",
                        fontsize=8, color="#f57f17" if is_n1 else "#333333",
                        fontweight="bold" if is_n1 else "normal")

            ax.set_ylim(0, 0.88)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(STAGES, fontsize=9)
            ax.grid(axis="y", zorder=0, alpha=0.5)
            ax.tick_params(axis="both", length=3)

            if row == 0:
                ax.set_title(col_titles[col], color=model_color,
                             fontsize=10, fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel(f"{nf} Features\nF1 Score", fontsize=10)

    plt.tight_layout(h_pad=2.5, w_pad=1.5)
    path = "figures/ds_fig4_perclass_grid.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
#  FIGURE 5 — MacroF1 preservation radar / line comparison
# ============================================================================
def fig5_macrof1_lines():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "MacroF1 Preservation — QGAN vs Classical GAN Augmentation\n"
        "QGANs maintain closer MacroF1 to baseline at all feature counts",
        fontsize=13, fontweight="bold", y=1.01
    )

    # Left: absolute MacroF1
    ax = axes[0]
    for model, label, color, ls in zip(
            MODELS, LABELS_SHORT, COLORS,
            ["-", "--", "-", "-"]):
        vals = [get(nf, model, "macro_f1") for nf in FEATS]
        lw = 2.5 if model == "baseline" else 2.0
        ax.plot(FEATS, vals, color=color, linewidth=lw,
                linestyle=ls, marker="o", markersize=7,
                label=label, zorder=4)
        for nf, v in zip(FEATS, vals):
            ax.annotate(f"{v:.3f}", (nf, v),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8.5, color=color)

    ax.set_title("Absolute Macro F1", pad=8)
    ax.set_xlabel("Number of EEG Features")
    ax.set_ylabel("Macro F1 Score")
    ax.set_xticks(FEATS)
    ax.set_ylim(0.28, 0.52)
    ax.grid(zorder=0)
    ax.legend(loc="lower right", handlelength=1.5)

    # Right: ΔMacroF1 (only augmented models)
    ax2 = axes[1]
    for model, label, color in zip(MODELS[1:], LABELS_SHORT[1:], COLORS[1:]):
        vals = [get(nf, model, "f1_improvement") for nf in FEATS]
        ax2.plot(FEATS, vals, color=color, linewidth=2.0,
                 marker="s", markersize=7, label=label, zorder=4)
        for nf, v in zip(FEATS, vals):
            ax2.annotate(f"{v:+.3f}", (nf, v),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8.5, color=color)

    ax2.axhline(0, color="#333333", linewidth=1.2,
                linestyle="--", zorder=5, label="No change")
    ax2.axhspan(-0.04, 0, alpha=0.06, color="red",   zorder=0)
    ax2.axhspan(0, 0.02, alpha=0.06, color="green", zorder=0)
    ax2.set_title("ΔMacro F1 vs Baseline", pad=8)
    ax2.set_xlabel("Number of EEG Features")
    ax2.set_ylabel("ΔMacro F1")
    ax2.set_xticks(FEATS)
    ax2.grid(zorder=0)
    ax2.legend(loc="lower right", handlelength=1.5)

    plt.tight_layout()
    path = "figures/ds_fig5_macrof1_lines.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
#  FIGURE 6 — Summary 4-panel (best single figure for paper inclusion)
# ============================================================================
def fig6_summary_4panel():
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    fig.suptitle(
        "Downstream Evaluation Summary — ANPHY-Sleep (EPCTL01–03, 2730 epochs)\n"
        "All generative models use WGAN-GP | RBF-SVM classifier | Real PSG labels",
        fontsize=14, fontweight="bold"
    )

    # ── Panel A: Accuracy grouped bar ────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    x, w = np.arange(len(FEATS)), 0.19
    for i, (model, label, color) in enumerate(zip(MODELS, LABELS_SHORT, COLORS)):
        vals   = [get(nf, model, "accuracy") for nf in FEATS]
        offset = (i - 1.5) * w
        ax_a.bar(x + offset, vals, w, label=label,
                 color=color, alpha=0.88, zorder=3, edgecolor="white", lw=0.6)
    ax_a.set_title("(A)  Accuracy  ↑", pad=6)
    ax_a.set_xticks(x); ax_a.set_xticklabels([f"{n}f" for n in FEATS])
    ax_a.set_ylim(0, 0.70); ax_a.grid(axis="y", zorder=0)
    ax_a.set_ylabel("Accuracy")
    ax_a.legend(loc="upper left", fontsize=8.5, ncol=2)

    # ── Panel B: N1 F1 grouped bar ───────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    for i, (model, label, color) in enumerate(zip(MODELS, LABELS_SHORT, COLORS)):
        vals   = [get_f1(nf, model, "N1") for nf in FEATS]
        offset = (i - 1.5) * w
        bars = ax_b.bar(x + offset, vals, w, label=label,
                        color=color, alpha=0.88, zorder=3, edgecolor="white", lw=0.6)
        # Mark collapses
        for bar, v, nf in zip(bars, vals, FEATS):
            if v == 0.0 and model != "baseline":
                ax_b.text(bar.get_x() + bar.get_width() / 2, 0.005,
                          "✗", ha="center", va="bottom",
                          fontsize=10, color="#c62828", fontweight="bold")
            elif v > 0:
                ax_b.text(bar.get_x() + bar.get_width() / 2,
                          v + 0.003, f"{v:.3f}",
                          ha="center", va="bottom", fontsize=7, color=color)

    ax_b.set_title("(B)  N1 Minority-Class F1  ↑  [KEY FINDING]", pad=6)
    ax_b.set_xticks(x); ax_b.set_xticklabels([f"{n}f" for n in FEATS])
    ax_b.set_ylim(0, 0.38); ax_b.grid(axis="y", zorder=0)
    ax_b.set_ylabel("N1 F1 Score")

    # Annotation arrow
    ax_b.annotate("Classical GAN\nalways 0.000 →",
                  xy=(0.28, 0.01), fontsize=8.5, color="#c62828",
                  style="italic", xycoords="axes fraction")

    # ── Panel C: ΔAccuracy line plot ─────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    for model, label, color in zip(MODELS[1:], LABELS_SHORT[1:], COLORS[1:]):
        vals = [get(nf, model, "acc_improvement") for nf in FEATS]
        ax_c.plot(FEATS, vals, color=color, linewidth=2.2,
                  marker="o", markersize=8, label=label, zorder=4)
        for nf, v in zip(FEATS, vals):
            ax_c.annotate(f"{v:+.3f}", (nf, v),
                          textcoords="offset points", xytext=(0, 9),
                          ha="center", fontsize=8.5, color=color)
    ax_c.axhline(0, color="#555555", lw=1.2, ls="--", zorder=5)
    ax_c.axhspan(-0.01, 0.085, alpha=0.04, color="green", zorder=0)
    ax_c.set_title("(C)  ΔAccuracy vs Baseline  ↑", pad=6)
    ax_c.set_xticks(FEATS)
    ax_c.set_xlabel("Number of EEG Features")
    ax_c.set_ylabel("ΔAccuracy")
    ax_c.grid(zorder=0)
    ax_c.legend(loc="upper left", fontsize=9)

    # ── Panel D: ΔMacroF1 line plot ──────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    for model, label, color in zip(MODELS[1:], LABELS_SHORT[1:], COLORS[1:]):
        vals = [get(nf, model, "f1_improvement") for nf in FEATS]
        ax_d.plot(FEATS, vals, color=color, linewidth=2.2,
                  marker="s", markersize=8, label=label, zorder=4)
        for nf, v in zip(FEATS, vals):
            ax_d.annotate(f"{v:+.3f}", (nf, v),
                          textcoords="offset points", xytext=(0, 9),
                          ha="center", fontsize=8.5, color=color)
    ax_d.axhline(0, color="#555555", lw=1.2, ls="--", zorder=5,
                 label="No change from baseline")
    ax_d.axhspan(-0.035, 0, alpha=0.06, color="red",   zorder=0)
    ax_d.axhspan(0, 0.002, alpha=0.06, color="green", zorder=0)
    ax_d.set_title("(D)  ΔMacro F1 vs Baseline  ↑", pad=6)
    ax_d.set_xticks(FEATS)
    ax_d.set_xlabel("Number of EEG Features")
    ax_d.set_ylabel("ΔMacro F1")
    ax_d.grid(zorder=0)
    ax_d.legend(loc="lower right", fontsize=9)

    path = "figures/ds_fig6_summary_4panel.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
#  FIGURE 7 — Accuracy heat-map table (visual table — easy to read in a talk)
# ============================================================================
def fig7_heatmap_table():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle(
        "Colour-Coded Results Table — ANPHY-Sleep Downstream Evaluation\n"
        "Green = best in group  |  Red = worst in group",
        fontsize=13, fontweight="bold", y=1.02
    )

    metric_pairs = [("Accuracy", "accuracy"), ("Macro F1", "macro_f1")]

    for ax, (metric, key) in zip(axes, metric_pairs):
        # Build matrix: rows = models, cols = features
        mat = np.array([[get(nf, m, key) for nf in FEATS] for m in MODELS])

        # Normalise per column for colour
        col_min = mat.min(axis=0, keepdims=True)
        col_max = mat.max(axis=0, keepdims=True)
        norm_mat = (mat - col_min) / (col_max - col_min + 1e-9)

        im = ax.imshow(norm_mat, cmap="RdYlGn", aspect="auto",
                       vmin=0, vmax=1)

        # Annotate cells
        for r in range(len(MODELS)):
            for c in range(len(FEATS)):
                v = mat[r, c]
                is_best  = (v == mat[:, c].max())
                is_worst = (v == mat[:, c].min())
                txt = f"{v:.4f}"
                if is_best:  txt += "\n★"
                fc = "white" if norm_mat[r, c] < 0.4 or norm_mat[r, c] > 0.75 else "#111111"
                ax.text(c, r, txt, ha="center", va="center",
                        fontsize=10, color=fc,
                        fontweight="bold" if is_best else "normal")

        ax.set_xticks(range(len(FEATS)))
        ax.set_xticklabels([f"{n} Features" for n in FEATS], fontsize=10)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(LABELS_SHORT, fontsize=10)
        ax.set_title(f"{metric}", pad=8)
        ax.tick_params(length=0)

        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        cb.set_label("Normalised score within feature group", fontsize=8)

    plt.tight_layout()
    path = "figures/ds_fig7_heatmap_table.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
#  CORRECTED LaTeX TABLE
# ============================================================================
def latex_table():
    lines = []
    lines.append("% ============================================================")
    lines.append("% Downstream SVM Evaluation — ANPHY-Sleep (EPCTL01-03)")
    lines.append("% CORRECTIONS:")
    lines.append("%   1. Caption: minority class is N1 (8.2%), NOT Wake (10.9%)")
    lines.append("%   2. Features=4: baseline MacroF1 0.4343 is best → bold")
    lines.append("% ============================================================\n")
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(
        r"\caption{Downstream sleep-stage classification on ANPHY-Sleep "
        r"(EPCTL01--03, 2\,730 epochs, 3 subjects) using real PSG stage labels "
        r"(W, N1, N2, N3, REM). All generative models use WGAN-GP for "
        r"architectural fairness; the sole variable is the generator type "
        r"(classical MLP vs.\ parameterised quantum circuit). "
        r"Each augments the minority class \textbf{N1} (8.2\,\% of epochs) "  # ← FIX 1
        r"with synthetic samples. "
        r"Classifier: RBF-SVM with balanced class weights. "
        r"\textbf{Bold} = best value per metric per feature group. "
        r"$\Delta$ = change over real-data-only baseline. "
        r"N1~F1 $= 0.000$ for Classical GAN across all feature counts "
        r"(minority-class collapse).}"
    )
    lines.append(r"\label{tab:downstream}")
    lines.append(r"\begin{tabular}{|c|l|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Feat.} & \textbf{Training Data} & "
        r"\textbf{Acc\,$\uparrow$} & $\boldsymbol{\Delta}$\textbf{Acc} & "
        r"\textbf{Macro F1\,$\uparrow$} & $\boldsymbol{\Delta}$\textbf{F1} & "
        r"\textbf{N1 F1\,$\uparrow$} \\"   # ← ADD N1 F1 column
    )
    lines.append(r"\hline")

    row_labels = {
        "baseline":           r"Real Only (Baseline)",
        "classical_gan_aug":  r"Real + Classical GAN (WGAN-GP)",
        "qgan_noiseless_aug": r"Real + QGAN Noiseless",
        "qgan_noisy_aug":     r"Real + QGAN Noisy",
    }

    for nf in FEATS:
        best_acc = max(get(nf, m, "accuracy")  for m in MODELS)
        best_f1  = max(get(nf, m, "macro_f1")  for m in MODELS)
        best_n1  = max(get_f1(nf, m, "N1")     for m in MODELS)

        for j, model in enumerate(MODELS):
            acc  = get(nf, model, "accuracy")
            mf1  = get(nf, model, "macro_f1")
            n1   = get_f1(nf, model, "N1")

            def bold(v, best, fmt=".4f"):
                s = f"{v:{fmt}}"
                return f"\\textbf{{{s}}}" if abs(v - best) < 1e-6 else s

            acc_s = bold(acc, best_acc)
            f1_s  = bold(mf1, best_f1)
            n1_s  = bold(n1,  best_n1)

            # Red text for collapse
            if n1 == 0.0 and model != "baseline":
                n1_s = r"\textcolor{red}{0.0000}"

            if model == "baseline":
                da_s = "---"
                df_s = "---"
            else:
                da = get(nf, model, "acc_improvement")
                df = get(nf, model, "f1_improvement")
                da_s = f"{da:+.4f}"
                df_s = f"{df:+.4f}"

            # Multirow feat number on first row only
            feat_col = f"\\multirow{{4}}{{*}}{{{nf}}}" if j == 0 else ""

            lines.append(
                f"  {feat_col} & {row_labels[model]} & "
                f"{acc_s} & {da_s} & {f1_s} & {df_s} & {n1_s} \\\\"
            )

        lines.append(r"  \hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1pt}")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize $^\dagger$\textcolor{red}{Red} = minority-class collapse (N1 F1 = 0). ")
    lines.append(r"Classical GAN collapses N1 F1 in all nine conditions.")
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append("% NOTE: add \\usepackage{xcolor} and \\usepackage{multirow} to preamble")

    out = "downstream_table_corrected.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  ✓ {out}")
    print()
    print("  CORRECTIONS APPLIED:")
    print("  1. Caption: 'minority class N1 (8.2%)' — NOT Wake")
    print("  2. Features=4: Baseline MacroF1 0.4343 bolded (it IS the best)")
    print("  3. Added N1 F1 column — the key quantum advantage finding")
    print("  4. Red colour for Classical GAN N1 collapse cells")
    print("  5. \\multirow for feature number, \\arraystretch for spacing")
    print("  6. Added xcolor/multirow package note")


# ============================================================================
#  MAIN
# ============================================================================
def main():
    print("\n  Generating publication-quality downstream figures...")
    print(f"  {'='*60}")

    fig1_overview()
    fig2_delta()
    fig3_n1_collapse()
    fig4_perclass_grid()
    fig5_macrof1_lines()
    fig6_summary_4panel()
    fig7_heatmap_table()
    latex_table()

    print(f"  {'='*60}")
    print(f"\n  All figures saved to figures/")
    print(f"  Corrected LaTeX saved to downstream_table_corrected.txt\n")

    # Quick terminal summary
    print(f"  {'─'*72}")
    print(f"  {'Feat':<5} {'Model':<32} {'Acc':<7} {'MacF1':<8} {'N1 F1':<8}")
    print(f"  {'─'*72}")
    short_names = {
        "baseline":           "Real Only (Baseline)          ",
        "classical_gan_aug":  "Real + Classical GAN (WGAN-GP)",
        "qgan_noiseless_aug": "Real + QGAN Noiseless         ",
        "qgan_noisy_aug":     "Real + QGAN Noisy             ",
    }
    for nf in FEATS:
        for model in MODELS:
            acc  = get(nf, model, "accuracy")
            mf1  = get(nf, model, "macro_f1")
            n1   = get_f1(nf, model, "N1")
            flag = "  ← COLLAPSE" if (n1 == 0.0 and model != "baseline") else ""
            print(f"  {str(nf)+'f':<5} {short_names[model]} "
                  f"{acc:<7.4f} {mf1:<8.4f} {n1:.4f}{flag}")
        print(f"  {'─'*72}")


if __name__ == "__main__":
    main()