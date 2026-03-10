# qgan/plot_final.py
# Generates the FINAL paper figures using all 3 results:
#   Classical GAN (CPU) + QGAN (CPU-Sim) + QGAN (QPU-Sim)
# Run AFTER both train.py and train_qpu.py complete:
#   python -m qgan.plot_final

import json
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

DARK   = "#0d1117"
BLUE   = "#4C9BE8"
ORANGE = "#E8834C"
GREEN  = "#4CE87A"
RED    = "#E84C4C"
GRID   = "#2a2a3a"
TEXT   = "#e0e0e0"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": GRID, "legend.labelcolor": TEXT,
    "text.color": TEXT, "grid.color": GRID,
    "grid.linestyle": "--", "grid.alpha": 0.4,
})

def save(name):
    path = f"figures/{name}"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  Saved: {path}")


def load_data():
    with open("results.json") as f:
        cpu_all = json.load(f)
    # use 4-feature experiment
    cpu_exp = next(r for r in cpu_all if r["n_features"] == 4)

    with open("results_qpu.json") as f:
        qpu = json.load(f)

    return cpu_exp, qpu


# ── Figure 1: Complete timing graphic ─────────────────────────────────────────
def fig_complete_timing(cpu_exp, qpu):
    c_time  = cpu_exp["classical"]["history"]["avg_time"]
    qc_time = cpu_exp["qgan"]["history"]["avg_time"]
    qq_time = qpu["history"]["avg_time"]

    labels = ["Classical GAN\n(CPU)", "QGAN\n(CPU-Sim)", "QGAN\n(QPU-Sim)"]
    values = [c_time, qc_time, qq_time]
    colors = [ORANGE, BLUE, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Epoch Training Time — Classical GAN vs QGAN-CPU vs QGAN-QPU",
                 fontsize=13, fontweight="bold")

    # absolute
    ax = axes[0]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor=GRID)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values)*0.01,
                f"{val:.1f}s", ha="center", fontweight="bold", fontsize=11)
    ax.set_title("Average Epoch Time (seconds)", fontweight="bold")
    ax.set_ylabel("Time (s)")
    ax.grid(True, axis="y")

    # log scale
    ax2 = axes[1]
    bars2 = ax2.bar(labels, values, color=colors, width=0.5, edgecolor=GRID)
    for bar, val in zip(bars2, values):
        ax2.text(bar.get_x() + bar.get_width()/2, val * 1.3,
                 f"{val:.1f}s", ha="center", fontweight="bold", fontsize=11)
    ax2.set_yscale("log")
    ax2.set_title("Epoch Time — Log Scale", fontweight="bold")
    ax2.set_ylabel("Time (s) — log scale")
    ax2.grid(True, axis="y")

    if c_time > 0:
        ratio_cpu = round(qc_time / c_time)
        ratio_qpu = round(qc_time / qq_time, 1) if qq_time > 0 else "N/A"
        ax2.annotate(f"QGAN-CPU\n~{ratio_cpu}× slower",
                     xy=(1, qc_time), xytext=(1.4, qc_time * 0.4),
                     color=TEXT, fontsize=9,
                     arrowprops=dict(arrowstyle="->", color=TEXT))
        if isinstance(ratio_qpu, float):
            ax2.annotate(f"QPU ~{ratio_qpu}×\nfaster than CPU-Sim",
                         xy=(2, qq_time), xytext=(1.5, qq_time * 3),
                         color=GREEN, fontsize=9,
                         arrowprops=dict(arrowstyle="->", color=GREEN))

    plt.tight_layout()
    save("fig_final_timing.png")


# ── Figure 2: Complete metrics table as bar chart ─────────────────────────────
def fig_complete_metrics(cpu_exp, qpu):
    c_clf  = cpu_exp["classical"]["clf"]
    qc_clf = cpu_exp["qgan"]["clf"]
    qq_clf = qpu["clf"]

    metrics = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1"]
    x = np.arange(len(metrics))
    w = 0.26

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Classification Metrics — Classical GAN vs QGAN-CPU vs QGAN-QPU (4 features)",
                 fontsize=13, fontweight="bold")

    vals_c  = [c_clf[m]  for m in metrics]
    vals_qc = [qc_clf[m] for m in metrics]
    vals_qq = [qq_clf[m] for m in metrics]

    b1 = ax.bar(x - w,   vals_c,  w, label="Classical GAN (CPU)",  color=ORANGE, alpha=0.9)
    b2 = ax.bar(x,       vals_qc, w, label="QGAN (CPU-Sim)",        color=BLUE,   alpha=0.9)
    b3 = ax.bar(x + w,   vals_qq, w, label="QGAN (QPU-Sim)",        color=GREEN,  alpha=0.9)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.2); ax.set_ylabel("Score")
    ax.axhline(0.5, color=RED, ls=":", lw=1, alpha=0.5, label="Random baseline")
    ax.grid(True, axis="y"); ax.legend(fontsize=10)
    plt.tight_layout()
    save("fig_final_metrics.png")


# ── Figure 3: Feature sweep with QPU included ─────────────────────────────────
def fig_feature_sweep_complete(cpu_exp, qpu):
    with open("results.json") as f:
        cpu_all = json.load(f)

    n_list  = [r["n_features"] for r in cpu_all]
    q_std   = [r["qgan"]["history"]["std_MAE"][-1]       for r in cpu_all]
    c_std   = [r["classical"]["history"]["std_MAE"][-1]  for r in cpu_all]
    q_mean  = [r["qgan"]["history"]["mean_MAE"][-1]      for r in cpu_all]
    c_mean  = [r["classical"]["history"]["mean_MAE"][-1] for r in cpu_all]
    times_q = [r["qgan"]["history"]["avg_time"]          for r in cpu_all]
    times_c = [r["classical"]["history"]["avg_time"]     for r in cpu_all]

    # add QPU point at 4 features
    qpu_std_mae  = qpu["history"]["std_MAE"][-1]  if qpu["history"]["std_MAE"]  else None
    qpu_mean_mae = qpu["history"]["mean_MAE"][-1] if qpu["history"]["mean_MAE"] else None
    qpu_time     = qpu["history"]["avg_time"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Optimization Analysis — Feature Sweep with QPU Comparison",
                 fontsize=13, fontweight="bold")

    # Std MAE
    ax = axes[0]
    ax.plot(n_list, q_std,  color=BLUE,   lw=2, marker="o", label="QGAN CPU-Sim")
    ax.plot(n_list, c_std,  color=ORANGE, lw=2, marker="s", ls="--", label="Classical GAN")
    if qpu_std_mae:
        ax.scatter([4], [qpu_std_mae], color=GREEN, s=150, zorder=5, label="QGAN QPU-Sim", marker="*")
    ax.set_title("Std MAE (↓ better — variance match)", fontweight="bold")
    ax.set_xlabel("Features"); ax.set_ylabel("Std MAE")
    ax.set_xticks(n_list); ax.grid(True); ax.legend()

    # Mean MAE
    ax2 = axes[1]
    ax2.plot(n_list, q_mean, color=BLUE,   lw=2, marker="o", label="QGAN CPU-Sim")
    ax2.plot(n_list, c_mean, color=ORANGE, lw=2, marker="s", ls="--", label="Classical GAN")
    if qpu_mean_mae:
        ax2.scatter([4], [qpu_mean_mae], color=GREEN, s=150, zorder=5, label="QGAN QPU-Sim", marker="*")
    ax2.set_title("Mean MAE (↓ better — mean match)", fontweight="bold")
    ax2.set_xlabel("Features"); ax2.set_ylabel("Mean MAE")
    ax2.set_xticks(n_list); ax2.grid(True); ax2.legend()

    # Timing
    ax3 = axes[2]
    ax3.plot(n_list, times_q, color=BLUE,   lw=2, marker="o", label="QGAN CPU-Sim")
    ax3.plot(n_list, times_c, color=ORANGE, lw=2, marker="s", ls="--", label="Classical GAN")
    ax3.scatter([4], [qpu_time], color=GREEN, s=150, zorder=5, label="QGAN QPU-Sim", marker="*")
    ax3.set_title("Avg Epoch Time (s)", fontweight="bold")
    ax3.set_xlabel("Features"); ax3.set_ylabel("Time (s)")
    ax3.set_xticks(n_list); ax3.grid(True); ax3.legend()

    plt.tight_layout()
    save("fig_final_sweep.png")


# ── Figure 4: Summary table as visual ─────────────────────────────────────────
def fig_summary_table(cpu_exp, qpu):
    c_clf  = cpu_exp["classical"]["clf"]
    qc_clf = cpu_exp["qgan"]["clf"]
    qq_clf = qpu["clf"]

    c_time  = cpu_exp["classical"]["history"]["avg_time"]
    qc_time = cpu_exp["qgan"]["history"]["avg_time"]
    qq_time = qpu["history"]["avg_time"]

    rows = [
        ["Classical GAN", "CPU",
         c_clf["Accuracy"], c_clf["Precision"],
         c_clf["Sensitivity"], c_clf["Specificity"], c_clf["F1"],
         f"{c_time:.2f}s"],
        ["QGAN", "CPU-Sim",
         qc_clf["Accuracy"], qc_clf["Precision"],
         qc_clf["Sensitivity"], qc_clf["Specificity"], qc_clf["F1"],
         f"{qc_time:.2f}s"],
        ["QGAN", "QPU-Sim",
         qq_clf["Accuracy"], qq_clf["Precision"],
         qq_clf["Sensitivity"], qq_clf["Specificity"], qq_clf["F1"],
         f"{qq_time:.2f}s"],
    ]
    cols = ["Model", "Hardware", "Accuracy", "Precision",
            "Sensitivity", "Specificity", "F1", "Epoch Time"]

    fig, ax = plt.subplots(figsize=(14, 3))
    fig.suptitle("Complete Results Table — Classical GAN / QGAN-CPU / QGAN-QPU (4 features)",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

    cell_colors = []
    for row in rows:
        row_colors = ["#1a1a2e"] * len(cols)
        cell_colors.append(row_colors)

    table = ax.table(
        cellText   = rows,
        colLabels  = cols,
        cellLoc    = "center",
        loc        = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # style header
    for j in range(len(cols)):
        table[0, j].set_facecolor("#1F4E79")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # style rows
    row_colors_map = [ORANGE + "33", BLUE + "33", GREEN + "33"]
    for i, rc in enumerate(row_colors_map):
        for j in range(len(cols)):
            table[i+1, j].set_facecolor("#1a1a2e")
            table[i+1, j].set_text_props(color=TEXT)

    plt.tight_layout()
    save("fig_final_table.png")


def main():
    print("\n  Generating final paper figures...")

    try:
        cpu_exp, qpu = load_data()
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print("  Make sure both train.py and train_qpu.py have completed.")
        return

    fig_complete_timing(cpu_exp, qpu)
    fig_complete_metrics(cpu_exp, qpu)
    fig_feature_sweep_complete(cpu_exp, qpu)
    fig_summary_table(cpu_exp, qpu)

    print("\n  Final figures saved to ./figures/")
    print("  fig_final_timing.png  — epoch time: Classical / QGAN-CPU / QGAN-QPU")
    print("  fig_final_metrics.png — all 5 metrics side by side")
    print("  fig_final_sweep.png   — feature sweep with QPU point added")
    print("  fig_final_table.png   — visual results table for paper")


if __name__ == "__main__":
    main()