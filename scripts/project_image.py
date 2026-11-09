"""Project image for the submission card (1600 x 900 PNG). Every number is read from manuscript/numbers.tex.

    cd analysis && uv run python ../scripts/project_image.py
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (ROOT / "manuscript" / "numbers.tex").read_text()))

SURFACE, PANEL, INK, INK2, MUTED = "#1a1a19", "#242422", "#f0efec", "#c3c2b7", "#8d8c85"
RECEIPT, TRANSCRIPT = "#3987e5", "#d95926"   # categorical slots 1 and 2 (dark mode)
plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def pct(k, n):
    return 100 * int(N[k]) / int(N[n])


def main(out=ROOT / "manuscript" / "figures" / "project_image.png"):
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1600)
    ax.set_ylim(900, 0)
    ax.axis("off")

    # header
    ax.text(80, 88, "APART SWARM ALIGNMENT HACKATHON 2026  ·  TRACK 4: MULTI-AGENT OVERSIGHT", color=MUTED,
            fontsize=15, fontweight="bold", va="center")
    ax.text(80, 170, "Receipts, Not Transcripts", color=INK, fontsize=58, fontweight="bold", va="center")
    ax.text(82, 240, "Overseeing agent swarms that can edit their own logs", color=INK2, fontsize=26, va="center")

    # mechanism: six agents, one shared cache, two records of what happened
    for i in range(6):
        x = 135 + i * 72
        ax.add_patch(Circle((x, 365), 24, facecolor=PANEL, edgecolor=INK2, lw=2))
        ax.text(x, 365, f"A{i + 1}", color=INK, fontsize=13, ha="center", va="center", fontweight="bold")
        ax.add_patch(FancyArrowPatch((x, 392), (315, 462), arrowstyle="-", color=MUTED, lw=1.2, alpha=0.7))
    ax.add_patch(FancyBboxPatch((195, 462), 240, 64, boxstyle="round,pad=0,rounding_size=12",
                                facecolor=PANEL, edgecolor=INK2, lw=2))
    ax.text(315, 494, "shared cache", color=INK, fontsize=18, ha="center", va="center")
    for (x0, x1, col) in ((260, 190, RECEIPT), (370, 440, TRANSCRIPT)):
        ax.add_patch(FancyArrowPatch((x0, 528), (x1, 598), arrowstyle="-|>", mutation_scale=18, color=col, lw=2.4))
    ax.add_patch(FancyBboxPatch((80, 600), 220, 112, boxstyle="round,pad=0,rounding_size=12",
                                facecolor=PANEL, edgecolor=RECEIPT, lw=2.5))
    ax.text(190, 638, "host receipts", color=INK, fontsize=19, ha="center", va="center", fontweight="bold")
    ax.text(190, 676, "agents cannot edit", color=RECEIPT, fontsize=15, ha="center", va="center")
    ax.add_patch(FancyBboxPatch((330, 600), 220, 112, boxstyle="round,pad=0,rounding_size=12",
                                facecolor=PANEL, edgecolor=TRANSCRIPT, lw=2.5, linestyle=(0, (6, 4))))
    ax.text(440, 638, "transcripts", color=INK, fontsize=19, ha="center", va="center", fontweight="bold")
    ax.text(440, 676, "agents can edit", color=TRANSCRIPT, fontsize=15, ha="center", va="center")
    ax.text(315, 760, "compare the two: what can an overseer conclude?", color=INK2, fontsize=15,
            ha="center", va="center", style="italic")

    # three findings, numbers from numbers.tex
    nat_no = pct("NatHeldAlarmNoTransfer", "NatHeldNoTransfer")
    nat_yes = pct("NatHeldAlarmTransfer", "NatHeldTransfers")
    tiles = [
        (f"{N['EfourABConsistencyVOneHiddenK']}/{N['EfourABConsistencyVOneHiddenN']}",
         "hidden transfers caught when the harness",
         "writes the transcript; no false alarm", RECEIPT),
        (f"{N['EfourABagentUnloggedCrossReadVOneHiddenK']}/{N['EfourABagentUnloggedCrossReadVOneHiddenN']}",
         "caught on agents' own logs by a cross-read",
         f"check; {N['EfouragentUnloggedCrossReadVOneNormalPct']}% and {N['EfourBagentUnloggedCrossReadVOneNormalPct']}% "
         "false alarms on A and B", RECEIPT),
        (f"{nat_no:.0f}% vs {nat_yes:.0f}%",
         "natural traffic: it fires as often without",
         f"a transfer ({N['NatHeldAlarmNoTransfer']}/{N['NatHeldNoTransfer']}) as with one "
         f"({N['NatHeldAlarmTransfer']}/{N['NatHeldTransfers']})", TRANSCRIPT),
    ]
    for i, (value, line1, line2, col) in enumerate(tiles):
        y = 318 + i * 150
        ax.add_patch(FancyBboxPatch((640, y), 880, 128, boxstyle="round,pad=0,rounding_size=14",
                                    facecolor=PANEL, edgecolor="none"))
        ax.add_patch(FancyBboxPatch((640, y), 8, 128, boxstyle="round,pad=0,rounding_size=4",
                                    facecolor=col, edgecolor="none"))
        ax.text(675, y + 64, value, color=INK, fontsize=40, fontweight="bold", va="center")
        ax.text(1045, y + 46, line1, color=INK2, fontsize=16, va="center")
        ax.text(1045, y + 82, line2, color=INK2, fontsize=16, va="center")

    # footer
    ax.text(80, 840, "CacheBoard  ·  pre-registered held-out tests  ·  one-command rebuild", color=MUTED,
            fontsize=15, va="center")
    ax.text(1520, 840, "github.com/rahulanilnair/Swarm-Alignment--APART", color=MUTED, fontsize=15,
            va="center", ha="right")
    fig.savefig(out, dpi=100, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
