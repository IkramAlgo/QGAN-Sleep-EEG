# qgan/plot_arch.py
# Generates comparison figures for the architecture ablation study.
# Run AFTER train_arch.py: python -m qgan.plot_arch
#
# FIGURES GENERATED:
#   fig_arch_summary.png      — all archs × both losses at 4 features (main paper figure)
#   fig_arch_stdmae.png       — StdMAE comparison (quantum variance advantage)
#   fig_arch_specificity.png  — Specificity comparison (classification quality)
#   fig_arch_timing.png       — Training time comparison

import json
import os
import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style (same as all other figures) ─────────────────────────────────────────
DARK_BG  = "#0d1117"
BLUE     = "#4C9BE8"
ORANGE   = "#E8834C"
GREEN    = "#4CE87A"
PURPLE   = "#B04CE8"
TEAL     = "#4CE8D4"
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


def load_result(arch, loss_type):
    fname = f"results_arch_{arch}_{loss_type}.json"
    if not os.path.exists(fname):
        print(f"  WARNING: {fname} not found — skipping")
        return None
    with open(fname) as f:
        return json.load(f)


def load_all():
    """Load all available arch results + Arch A reference values."""
    results = {}

    # Arch A — existing results (hardcoded from results.json and results_wgan.json)
    results["A_bce"] = {
        "arch": "A", "loss_type": "bce",
        "StdMAE":   0.0797, "MeanMAE":  0.5247,
        "Acc":      0.5000, "Spec":     0.0000,
        "F1":       0.6667, "Time":     67.6,
        "label": "Arch A\n(Ring BCE)"
    }
    results["A_wgan"] = {
        "arch": "A", "loss_type": "wgan",
        "StdMAE":   0.0808, "MeanMAE":  0.4863,
        "Acc":      0.8250, "Spec":     0.7300,
        "F1":       0.8402, "Time":     14.6,
        "label": "Arch A\n(Ring WGAN-GP)"
    }

    # Load new arch results
    for arch in ["B", "C", "D"]:
        for loss in ["bce", "wgan"]:
            r = load_result(arch, loss)
            if r is not None:
                key = f"{arch}_{loss}"
                h   = r["history"]
                clf = r["clf"]
                loss_label = "WGAN-GP" if loss == "wgan" else "BCE"
                results[key] = {
                    "arch":      arch,
                    "loss_type": loss,
                    "StdMAE":    h["std_MAE"][-1],
                    "MeanMAE":   h["mean_MAE"][-1],
                    "Acc":       clf["Accuracy"],
                    "Spec":      clf["Specificity"],
                    "F1":        clf["F1"],
                    "Time":      h["avg_time"],
                    "label":     f"Arch {arch}\n({loss_label})",
                }

    return results


def fig_arch_summary(results):
    """
    Main paper figure: 4-panel bar chart comparing all architectures.
    Panels: StdMAE, Specificity, F1, Training Time
    """
    keys   = sorted(results.keys())
    labels = [results[k]["label"] for k in keys]
    colors = []
    for k in keys:
        if k.startswith("A"):
            colors.append(ORANGE if "bce" in k else TEAL)
        elif k.startswith("B"):
            colors.append(BLUE if "bce" in k else PURPLE)
        elif k.startswith("C"):
            colors.append("#E84C4C" if "bce" in k else "#E8A04C")
        elif k.startswith("D"):
            colors.append(GREEN if "bce" in k else "#4C8CE8")

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle(
        "Architecture Ablation Study — HybridQGAN (4 features)\n"
        "A=Ring  B=RZ+DualRot  C=6Qubits  D=All-to-All",
        fontsize=12, fontweight="bold"
    )

    metrics = [
        ("StdMAE",  "Std MAE ↓ (quantum variance advantage)", True),
        ("Spec",    "Specificity ↑ (classification quality)", False),
        ("F1",      "F1 Score ↑",                            False),
        ("Time",    "Avg Time/Epoch (s) ↓",                  True),
    ]

    x = np.arange(len(keys))
    for ax, (metric, title, lower_better) in zip(axes, metrics):
        vals = [results[k][metric] for k in keys]
        bars = ax.bar(x, vals, color=colors, width=0.65, edgecolor=GRID_COL)

        # label each bar
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

        # highlight best bar
        best_val = min(vals) if lower_better else max(vals)
        best_idx = vals.index(best_val)
        ax.get_children()[best_idx].set_edgecolor("gold")
        ax.get_children()[best_idx].set_linewidth(2.5)
        ax.text(x[best_idx], best_val * 0.5, "★ BEST",
                ha="center", color="gold", fontsize=8, fontweight="bold")

        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.grid(True, axis="y")
        ax.set_ylim(0, max(vals) * 1.25)

    plt.tight_layout()
    save("fig_arch_summary.png")


def fig_arch_stdmae(results):
    """
    Focused figure: StdMAE across architectures.
    Lower = better quantum variance matching.
    """
    keys   = [k for k in sorted(results.keys())]
    labels = [results[k]["label"] for k in keys]
    vals   = [results[k]["StdMAE"] for k in keys]
    colors = []
    for k in keys:
        if "bce"  in k: colors.append(BLUE)
        else:            colors.append(ORANGE)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        "StdMAE Comparison — Quantum Variance Advantage\n"
        "Lower = generator matches EEG variance distribution better",
        fontsize=12, fontweight="bold"
    )

    x    = np.arange(len(keys))
    bars = ax.bar(x, vals, color=colors, width=0.6, edgecolor=GRID_COL)

    # reference line: best existing (Arch A BCE = 0.0797)
    ax.axhline(0.0797, color="gold", lw=1.5, ls="--", alpha=0.7,
               label="Arch A BCE baseline (0.0797)")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    best_val = min(vals)
    best_idx = vals.index(best_val)
    ax.text(x[best_idx], best_val + 0.008, "★ BEST",
            ha="center", color="gold", fontsize=10, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BLUE,   label="BCE loss"),
        Patch(facecolor=ORANGE, label="WGAN-GP loss"),
    ]
    ax.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color="gold", ls="--", lw=1.5,
                   label="Arch A BCE baseline")
    ], fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Std MAE")
    ax.grid(True, axis="y")
    ax.set_ylim(0, max(vals) * 1.3)

    plt.tight_layout()
    save("fig_arch_stdmae.png")


def fig_arch_specificity(results):
    """
    Focused figure: Specificity across architectures.
    Specificity = ability to correctly reject fake data.
    Key finding: WGAN-GP fixes Spec=0 in Arch A — does it also fix B, C, D?
    """
    keys   = sorted(results.keys())
    labels = [results[k]["label"] for k in keys]
    vals   = [results[k]["Spec"] for k in keys]
    colors = [BLUE if "bce" in k else ORANGE for k in keys]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        "Specificity Comparison — Architecture × Loss Function\n"
        "Does WGAN-GP fix Spec=0 across all quantum generator architectures?",
        fontsize=12, fontweight="bold"
    )

    x    = np.arange(len(keys))
    bars = ax.bar(x, vals, color=colors, width=0.6, edgecolor=GRID_COL)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color="red" if val == 0.0 else TEXT_COL)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Specificity")
    ax.set_ylim(0, 1.2)
    ax.grid(True, axis="y")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=BLUE,   label="BCE loss"),
        Patch(facecolor=ORANGE, label="WGAN-GP loss"),
    ], fontsize=9)

    plt.tight_layout()
    save("fig_arch_specificity.png")


def print_comparison_table(results):
    """Print results table to terminal for the paper."""
    print(f"\n{'='*85}")
    print(f"  ARCHITECTURE ABLATION — COMPLETE RESULTS TABLE (4 features)")
    print(f"{'='*85}")
    print(f"  {'Architecture':<20} {'Loss':<10} {'MeanMAE':<10} {'StdMAE':<10} "
          f"{'Acc':<8} {'Spec':<8} {'F1':<8} {'Time/ep':<10}")
    print(f"  {'-'*81}")

    arch_names = {
        "A": "Ring CNOT (orig)",
        "B": "RZ+DualRot",
        "C": "6 Qubits",
        "D": "All-to-All",
    }
    loss_names = {"bce": "BCE", "wgan": "WGAN-GP"}

    for k in sorted(results.keys()):
        r = results[k]
        arch = arch_names.get(r["arch"], r["arch"])
        loss = loss_names.get(r["loss_type"], r["loss_type"])
        print(f"  {arch:<20} {loss:<10} {r['MeanMAE']:<10.4f} {r['StdMAE']:<10.4f} "
              f"{r['Acc']:<8.4f} {r['Spec']:<8.4f} {r['F1']:<8.4f} {r['Time']:.1f}s")

    print(f"{'='*85}\n")

    # Highlight key findings
    vals = results
    best_std  = min(vals, key=lambda k: vals[k]["StdMAE"])
    best_spec = max(vals, key=lambda k: vals[k]["Spec"])
    best_f1   = max(vals, key=lambda k: vals[k]["F1"])

    print(f"  KEY FINDINGS:")
    print(f"    Best StdMAE  : {vals[best_std]['label'].replace(chr(10),' ')} = {vals[best_std]['StdMAE']:.4f}")
    print(f"    Best Spec    : {vals[best_spec]['label'].replace(chr(10),' ')} = {vals[best_spec]['Spec']:.4f}")
    print(f"    Best F1      : {vals[best_f1]['label'].replace(chr(10),' ')} = {vals[best_f1]['F1']:.4f}")


def main():
    print("\n  Loading architecture ablation results...")
    results = load_all()

    if len(results) == 2:
        print("  No new arch results found yet.")
        print("  Run python -m qgan.train_arch first.")
        print("  (Arch A reference values loaded)")
        return

    print_comparison_table(results)

    print("\n  Generating figures...")
    fig_arch_summary(results)
    fig_arch_stdmae(results)
    fig_arch_specificity(results)

    print(f"\n  All architecture figures saved to ./{FIGURES_DIR}/")


if __name__ == "__main__":
    main()