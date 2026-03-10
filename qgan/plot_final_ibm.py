# qgan/plot_final_ibm.py
# Final paper figures using real IBM QPU results
# Reads: results.json (CPU) + results_ibm.json (QPU)
# Run: python -m qgan.plot_final_ibm

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


def load_all():
    with open("results.json") as f:
        cpu_all = json.load(f)
    cpu_exp = next(r for r in cpu_all if r["n_features"] == 4)
    with open("results_ibm.json") as f:
        ibm = json.load(f)
    return cpu_all, cpu_exp, ibm


def fig_epoch_times(cpu_exp, ibm):
    c_time  = cpu_exp["classical"]["history"]["avg_time"]
    qc_time = cpu_exp["qgan"]["history"]["avg_time"]
    qi_time = ibm["history"]["avg_time"]

    labels = ["Classical GAN\n(CPU)", "QGAN\n(CPU-Sim)", "QGAN\n(QPU-Sim, IBM 127q)"]
    values = [c_time, qc_time, qi_time]
    colors = [ORANGE, BLUE, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Figure 4: Epoch Training Time — Classical GAN / QGAN-CPU / QGAN-QPU",
        fontsize=13, fontweight="bold"
    )

    # linear scale
    ax = axes[0]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor=GRID, linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.1f}s", ha="center", fontsize=12, fontweight="bold")
    ax.set_title("Average Epoch Time (seconds)", fontweight="bold")
    ax.set_ylabel("Time (s)")
    ax.grid(True, axis="y", alpha=0.4)

    # log scale
    ax2 = axes[1]
    bars2 = ax2.bar(labels, values, color=colors, width=0.5, edgecolor=GRID, linewidth=1.2)
    for bar, val in zip(bars2, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.5,
                 f"{val:.1f}s", ha="center", fontsize=12, fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_title("Epoch Time — Log Scale (shows full gap)", fontweight="bold")
    ax2.set_ylabel("Time (s) — log scale")
    ax2.grid(True, axis="y", alpha=0.4)

    # FIXED: QGAN-CPU is slower than Classical
    if c_time > 0 and qc_time > 0:
        ratio_cpu = round(qc_time / c_time)
        ax2.annotate(
            f"QGAN-CPU is\n~{ratio_cpu}× slower\nthan Classical",
            xy=(1, qc_time), xytext=(1.35, qc_time * 0.3),
            color=TEXT, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.2)
        )

    # FIXED: QPU is SLOWER than CPU-Sim — was wrong before
    if qc_time > 0 and qi_time > 0:
        ratio_qpu = round(qi_time / qc_time)
        ax2.annotate(
            f"QGAN-QPU is\n~{ratio_qpu}× slower\nthan CPU-Sim",
            xy=(2, qi_time), xytext=(0.6, qi_time * 0.15),
            color=GREEN, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2)
        )

    plt.tight_layout()
    save("fig_ibm_timing.png")


def fig_all_metrics(cpu_exp, ibm):
    c_clf  = cpu_exp["classical"]["clf"]
    qc_clf = cpu_exp["qgan"]["clf"]
    qi_clf = ibm["clf"]

    metrics = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1"]
    x = np.arange(len(metrics))
    w = 0.26

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        "Figure 5: Classification Metrics — Classical GAN vs QGAN-CPU vs QGAN-QPU (4 features)",
        fontsize=13, fontweight="bold"
    )

    vals_c  = [c_clf[m]  for m in metrics]
    vals_qc = [qc_clf[m] for m in metrics]
    vals_qi = [qi_clf[m] for m in metrics]

    b1 = ax.bar(x - w, vals_c,  w, color=ORANGE, label="Classical GAN (CPU)",           alpha=0.9, edgecolor=DARK)
    b2 = ax.bar(x,     vals_qc, w, color=BLUE,   label="QGAN (CPU-Sim)",                 alpha=0.9, edgecolor=DARK)
    b3 = ax.bar(x + w, vals_qi, w, color=GREEN,  label="QGAN (QPU-Sim, IBM 127-qubit)",  alpha=0.9, edgecolor=DARK)

    for bars_group in [b1, b2, b3]:
        for bar in bars_group:
            h = bar.get_height()
            if h > 0.03:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score")
    ax.axhline(0.5, color=RED, ls=":", lw=1.2, alpha=0.6, label="Random baseline (0.5)")
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    save("fig_ibm_metrics.png")


def fig_loss_curves(cpu_exp, ibm):
    qc_g = cpu_exp["qgan"]["history"]["gen_loss"]
    qc_d = cpu_exp["qgan"]["history"]["disc_loss"]
    qi_g = ibm["history"]["gen_loss"]
    qi_d = ibm["history"]["disc_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Figure 6: Training Loss — QGAN CPU-Sim vs QGAN QPU-Sim",
        fontsize=13, fontweight="bold"
    )

    for ax, cpu_loss, qpu_loss, title in zip(
        axes,
        [qc_g, qc_d],
        [qi_g, qi_d],
        ["Generator Loss", "Discriminator Loss"]
    ):
        ax.plot(range(1, len(cpu_loss) + 1), cpu_loss,
                color=BLUE,  lw=2, label="QGAN CPU-Sim")
        ax.plot(range(1, len(qpu_loss) + 1), qpu_loss,
                color=GREEN, lw=2, ls="--", label="QGAN QPU-Sim (IBM 127-qubit)")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    save("fig_ibm_loss.png")


def fig_feature_sweep(cpu_all, ibm):
    n_list = [r["n_features"] for r in cpu_all]
    q_std  = [r["qgan"]["history"]["std_MAE"][-1]      for r in cpu_all]
    c_std  = [r["classical"]["history"]["std_MAE"][-1] for r in cpu_all]
    q_mean = [r["qgan"]["history"]["mean_MAE"][-1]      for r in cpu_all]
    c_mean = [r["classical"]["history"]["mean_MAE"][-1] for r in cpu_all]

    qi_std  = ibm["history"]["std_MAE"][-1]  if ibm["history"]["std_MAE"]  else None
    qi_mean = ibm["history"]["mean_MAE"][-1] if ibm["history"]["mean_MAE"] else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Figure 3: Distribution Matching — Feature Sweep with Real QPU Point",
        fontsize=13, fontweight="bold"
    )

    for ax, q_vals, c_vals, qi_val, title, ylabel in zip(
        axes,
        [q_std,  q_mean],
        [c_std,  c_mean],
        [qi_std, qi_mean],
        ["Std MAE (↓ better — variance matching)",
         "Mean MAE (↓ better — mean matching)"],
        ["Std MAE", "Mean MAE"]
    ):
        ax.plot(n_list, q_vals, color=BLUE,   lw=2.5, marker="o", label="QGAN CPU-Sim")
        ax.plot(n_list, c_vals, color=ORANGE, lw=2.5, marker="s", ls="--", label="Classical GAN")
        if qi_val is not None:
            ax.scatter([4], [qi_val], color=GREEN, s=200, zorder=6,
                       marker="*", label="QGAN QPU-Sim (IBM 127-qubit)")
            ax.annotate(f"  QPU: {qi_val:.4f}", xy=(4, qi_val), color=GREEN, fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Number of Features")
        ax.set_ylabel(ylabel)
        ax.set_xticks(n_list)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    save("fig_ibm_sweep.png")


# FIXED: hardware column now shows "QPU-Sim\n(IBM 127q)" — no truncation
def fig_summary_table(cpu_exp, ibm):
    c  = cpu_exp["classical"]
    qc = cpu_exp["qgan"]
    qi = ibm

    cols = ["Model", "Hardware", "Accuracy", "Precision",
            "Sensitivity", "Specificity", "F1 Score", "Epoch Time"]

    rows = [
        ["Classical GAN", "CPU",
         f"{c['clf']['Accuracy']:.4f}",  f"{c['clf']['Precision']:.4f}",
         f"{c['clf']['Sensitivity']:.4f}", f"{c['clf']['Specificity']:.4f}",
         f"{c['clf']['F1']:.4f}",         f"{c['history']['avg_time']:.2f}s"],

        ["QGAN", "CPU-Sim\n(noiseless)",
         f"{qc['clf']['Accuracy']:.4f}", f"{qc['clf']['Precision']:.4f}",
         f"{qc['clf']['Sensitivity']:.4f}", f"{qc['clf']['Specificity']:.4f}",
         f"{qc['clf']['F1']:.4f}",          f"{qc['history']['avg_time']:.2f}s"],

        ["QGAN", "QPU-Sim\n(IBM 127q)",
         f"{qi['clf']['Accuracy']:.4f}", f"{qi['clf']['Precision']:.4f}",
         f"{qi['clf']['Sensitivity']:.4f}", f"{qi['clf']['Specificity']:.4f}",
         f"{qi['clf']['F1']:.4f}",          f"{qi['history']['avg_time']:.2f}s"],
    ]

    fig, ax = plt.subplots(figsize=(16, 3.5))
    fig.patch.set_facecolor(DARK)
    fig.suptitle(
        "Table: Complete Results — Classical GAN / QGAN CPU-Sim / QGAN QPU-Sim (4 features)",
        fontsize=12, fontweight="bold"
    )
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.8)

    header_color = "#1a4a7a"
    row_colors   = ["#1e2a1e", "#1a1e2a", "#1e2a2a"]

    for j in range(len(cols)):
        tbl[0, j].set_facecolor(header_color)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(len(rows)):
        for j in range(len(cols)):
            tbl[i + 1, j].set_facecolor(row_colors[i])
            tbl[i + 1, j].set_text_props(color=TEXT)

    plt.tight_layout()
    save("fig_ibm_table.png")


def main():
    print("\n  Generating final IBM QPU paper figures...")

    try:
        cpu_all, cpu_exp, ibm = load_all()
    except FileNotFoundError as e:
        print(f"\n  Error: {e}")
        print("  Make sure both files exist:")
        print("    results.json     → run: python -m qgan.train")
        print("    results_ibm.json → run: python -m qgan.train_ibm")
        return

    print(f"  QPU device : {ibm['device']}")
    print(f"  QPU shots  : {ibm['shots']}")
    print(f"  QPU epochs : {ibm['epochs']}\n")

    fig_epoch_times(cpu_exp, ibm)
    fig_all_metrics(cpu_exp, ibm)
    fig_loss_curves(cpu_exp, ibm)
    fig_feature_sweep(cpu_all, ibm)
    fig_summary_table(cpu_exp, ibm)

    print("\n  ╔═══════════════════════════════════════════════╗")
    print("  ║  All final figures saved to ./figures/       ║")
    print("  ╠═══════════════════════════════════════════════╣")
    print("  ║  fig_ibm_timing.png  — epoch time comparison ║")
    print("  ║  fig_ibm_metrics.png — all 5 metrics         ║")
    print("  ║  fig_ibm_loss.png    — training loss curves  ║")
    print("  ║  fig_ibm_sweep.png   — feature sweep + QPU   ║")
    print("  ║  fig_ibm_table.png   — final results table   ║")
    print("  ╚═══════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()