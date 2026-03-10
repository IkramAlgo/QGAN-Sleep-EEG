"""
qgan/plot_qgan_tables.py
Generates QGAN results tables in same format as Sthefanie's paper.
Reads ALL values from results.json and results_ibm.json — nothing hardcoded.
Run: python -m qgan.plot_qgan_tables
"""

import json
import os
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

DARK     = "#0d1117"
HEADER   = "#1a4a7a"
ROW_C    = "#1e2a1e"
ROW_Q    = "#1a1e2a"
ROW_QPU  = "#1e2a2a"
TEXT     = "#e0e0e0"
WHITE    = "#ffffff"
GOLD     = "#ffd700"
NOTE_COL = "#888888"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "text.color": TEXT,
})

NOTE = (
    "† Single-run evaluation. No cross-validation. "
    "QPU-Sim tested at 4 features only due to computational constraints (1878s/epoch)."
)

COLS = ["Hardware", "Features", "Accuracy", "Precision",
        "Sensitivity", "Specificity", "F1 Score"]

COLS_COMBINED = ["Model / Hardware", "Features", "Accuracy", "Precision",
                 "Sensitivity", "Specificity", "F1 Score"]


def fmt(val):
    return f"{float(val):.4f}"


def clf_row(hardware, n_features, clf):
    return [
        hardware,
        str(n_features),
        fmt(clf["Accuracy"]),
        fmt(clf["Precision"]),
        fmt(clf["Sensitivity"]),
        fmt(clf["Specificity"]),
        fmt(clf["F1"]),
    ]


def load_data():
    with open("results.json") as f:
        cpu_all = json.load(f)
    cpu_all = sorted(cpu_all, key=lambda r: r["n_features"])
    with open("results_ibm.json") as f:
        ibm = json.load(f)
    return cpu_all, ibm


def apply_style(tbl, row_colors, col_count):
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.6)
    for j in range(col_count):
        cell = tbl[0, j]
        cell.set_facecolor(HEADER)
        cell.set_text_props(color=WHITE, fontweight="bold")
    for i, color in enumerate(row_colors):
        for j in range(col_count):
            cell = tbl[i + 1, j]
            cell.set_facecolor(color)
            cell.set_text_props(color=TEXT)


def save_fig(name):
    path = f"figures/{name}"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  Saved: {path}")


def table_classical(cpu_all):
    rows   = [clf_row("CPU", r["n_features"], r["classical"]["clf"]) for r in cpu_all]
    colors = [ROW_C] * len(rows)
    fig, ax = plt.subplots(figsize=(14, 1.2 + len(rows) * 0.7))
    fig.patch.set_facecolor(DARK)
    fig.suptitle("TABLE A\nCLASSICAL GAN — METRICS OF EVALUATION (CPU, Single Run) †",
                 fontsize=11, fontweight="bold", color=GOLD, y=1.04)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=COLS, cellLoc="center", loc="center")
    apply_style(tbl, colors, len(COLS))
    ax.text(0.0, -0.18, NOTE, transform=ax.transAxes, fontsize=8, color=NOTE_COL, style="italic")
    plt.tight_layout()
    save_fig("table_A_classical_gan.png")


def table_qgan_cpu(cpu_all):
    rows   = [clf_row("CPU-Sim (noiseless)", r["n_features"], r["qgan"]["clf"]) for r in cpu_all]
    colors = [ROW_Q] * len(rows)
    fig, ax = plt.subplots(figsize=(14, 1.2 + len(rows) * 0.7))
    fig.patch.set_facecolor(DARK)
    fig.suptitle("TABLE B\nQGAN CPU-SIM (Noiseless) — METRICS OF EVALUATION (Single Run) †",
                 fontsize=11, fontweight="bold", color=GOLD, y=1.04)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=COLS, cellLoc="center", loc="center")
    apply_style(tbl, colors, len(COLS))
    ax.text(0.0, -0.18, NOTE, transform=ax.transAxes, fontsize=8, color=NOTE_COL, style="italic")
    plt.tight_layout()
    save_fig("table_B_qgan_cpu.png")


def table_qgan_qpu(ibm):
    n_feat  = ibm.get("n_features", 4)
    device  = ibm.get("device", "QPU-Sim")
    shots   = ibm.get("shots", "?")
    rows = [clf_row(f"{device}\n({shots} shots)", n_feat, ibm["clf"])]
    colors  = [ROW_QPU]
    fig, ax = plt.subplots(figsize=(14, 2.2))
    fig.patch.set_facecolor(DARK)
    fig.suptitle("TABLE C\nQGAN QPU-SIM (IBM 127-Qubit Noise Model) — METRICS OF EVALUATION (Single Run) †",
                 fontsize=11, fontweight="bold", color=GOLD, y=1.06)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=COLS, cellLoc="center", loc="center")
    apply_style(tbl, colors, len(COLS))
    ax.text(0.0, -0.32, NOTE, transform=ax.transAxes, fontsize=8, color=NOTE_COL, style="italic")
    plt.tight_layout()
    save_fig("table_C_qgan_qpu.png")


def table_all_combined(cpu_all, ibm):
    rows   = []
    colors = []

    for r in cpu_all:
        rows.append(["Classical GAN / CPU"] + clf_row("", r["n_features"], r["classical"]["clf"])[1:])
        colors.append(ROW_C)

    for r in cpu_all:
        rows.append(["QGAN / CPU-Sim"] + clf_row("", r["n_features"], r["qgan"]["clf"])[1:])
        colors.append(ROW_Q)

    n_feat = ibm.get("n_features", 4)
    device = ibm.get("device", "QPU-Sim")
    rows.append([f"QGAN /\n{device}"] + clf_row("", n_feat, ibm["clf"])[1:])
    colors.append(ROW_QPU)

    fig, ax = plt.subplots(figsize=(16, 1.5 + len(rows) * 0.62))
    fig.patch.set_facecolor(DARK)
    fig.suptitle("TABLE D\nCOMPLETE RESULTS — Classical GAN / QGAN CPU-Sim / QGAN QPU-Sim (Single Run) †",
                 fontsize=11, fontweight="bold", color=GOLD, y=1.02)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=COLS_COMBINED, cellLoc="center", loc="center")
    apply_style(tbl, colors, len(COLS_COMBINED))
    ax.text(0.0, -0.06, NOTE, transform=ax.transAxes, fontsize=8, color=NOTE_COL, style="italic")
    plt.tight_layout()
    save_fig("table_D_all_models.png")


def main():
    print("\n  Reading results.json + results_ibm.json...")

    try:
        cpu_all, ibm = load_data()
    except FileNotFoundError as e:
        print(f"\n  Error: {e}")
        print("  Make sure both files exist:")
        print("    results.json     → run: python -m qgan.train")
        print("    results_ibm.json → run: python -m qgan.train_ibm")
        return

    features_found = [r["n_features"] for r in cpu_all]
    print(f"  CPU features found : {features_found}")
    print(f"  QPU device         : {ibm.get('device', '?')}")
    print(f"  QPU shots          : {ibm.get('shots', '?')}")
    print(f"  QPU epochs         : {ibm.get('epochs', '?')}\n")

    table_classical(cpu_all)
    table_qgan_cpu(cpu_all)
    table_qgan_qpu(ibm)
    table_all_combined(cpu_all, ibm)

    print("\n  ╔══════════════════════════════════════════════════╗")
    print("  ║  4 tables saved to ./figures/                   ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print("  ║  table_A_classical_gan.png  — Classical GAN     ║")
    print("  ║  table_B_qgan_cpu.png       — QGAN CPU-Sim      ║")
    print("  ║  table_C_qgan_qpu.png       — QGAN QPU-Sim      ║")
    print("  ║  table_D_all_models.png     — All 3 combined ★  ║")
    print("  ╚══════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()