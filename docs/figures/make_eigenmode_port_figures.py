"""Draw the geometry diagrams used by the eigenmode-port user guide.

Run from any directory with ``python docs/figures/make_eigenmode_port_figures.py``.
The dimensions in the straight-guide panel match Example 1. The horn layout
is schematic: the auxiliary guide is a separate computational grid.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


OUTPUT = Path(__file__).resolve().parents[1] / "source" / "_images" / "eigenmode"
BLUE = "#1f6f9e"
ORANGE = "#c46b17"
GREEN = "#208060"
PALE_BLUE = "#dcecf4"
PALE_GREY = "#e9edf0"


def arrow(axis, start, end, color, label=None, label_offset=(0, 0)):
    axis.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, lw=2, color=color)
    )
    if label:
        axis.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label,
            color=color,
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )


def straight_guide():
    fig, ax = plt.subplots(figsize=(10.5, 4.0), layout="constrained")
    ax.add_patch(Rectangle((0, 0), 240, 80, facecolor="white", edgecolor="#343b40", lw=1.5))
    ax.add_patch(Rectangle((0, 30), 240, 20, facecolor=PALE_BLUE, edgecolor="none"))
    for x, y, width, height in ((0, 0, 5, 80), (235, 0, 5, 80), (5, 0, 230, 5), (5, 75, 230, 5)):
        ax.add_patch(Rectangle((x, y), width, height, facecolor=PALE_GREY, edgecolor="none", alpha=0.8))
    ax.plot([20, 20], [5, 75], color=ORANGE, lw=3)
    ax.plot([235, 235], [5, 75], color=GREEN, lw=3)
    arrow(ax, (24, 40), (54, 40), ORANGE, "launch +x", (0, 3))
    arrow(ax, (231, 40), (201, 40), GREEN)
    ax.text(207, 46, "normal −x", ha="center", color=GREEN, fontsize=10, weight="bold")
    ax.text(120, 40, "dielectric core, εᵣ = 9", ha="center", va="center", color="#164663", weight="bold")
    ax.annotate("port 1\nx = 20 mm", (20, 73), (30, 88), color=ORANGE, ha="center",
                arrowprops={"arrowstyle": "-", "color": ORANGE})
    ax.annotate("port 2\nx = 235 mm", (235, 73), (210, 88), color=GREEN, ha="center",
                arrowprops={"arrowstyle": "-", "color": GREEN})
    ax.text(2.5, 18, "PML", rotation=90, ha="center", va="center", fontsize=9)
    ax.text(237.5, 18, "PML", rotation=90, ha="center", va="center", fontsize=9)
    ax.text(120, 68, "air cladding; modal aperture spans y = 5–75 mm", ha="center", fontsize=10)
    ax.set(xlim=(-2, 242), ylim=(-3, 97), xlabel="x (mm)", ylabel="y (mm)")
    ax.set_xticks([0, 20, 60, 120, 180, 240])
    ax.set_yticks([0, 5, 30, 50, 75, 80])
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Example 1: two ports on a uniform 2D dielectric guide", pad=20)
    fig.savefig(OUTPUT / "straight_waveguide_geometry.png", dpi=180)
    plt.close(fig)


def virtual_horn():
    fig, (main, auxiliary) = plt.subplots(
        2, 1, figsize=(10.5, 5.2), layout="constrained", gridspec_kw={"height_ratios": [1.45, 1]}
    )
    main.set(xlim=(0, 10), ylim=(0, 4))
    main.add_patch(Rectangle((0.3, 0.3), 9.4, 3.3, facecolor="white", edgecolor="#343b40", lw=1.4))
    main.add_patch(Rectangle((9.15, 0.3), 0.55, 3.3, facecolor=PALE_GREY, edgecolor="none"))
    main.add_patch(Rectangle((1.35, 0.75), 7.0, 2.5, fill=False, edgecolor=GREEN, lw=2, ls="--"))
    main.plot([1.35, 1.35], [0.75, 3.25], color=GREEN, lw=2)
    main.add_patch(Rectangle((2.35, 1.75), 1.4, 0.5, facecolor=PALE_BLUE, edgecolor=BLUE, lw=1.5))
    main.add_patch(plt.Polygon([(3.75, 1.75), (7.25, 0.95), (7.25, 3.05), (3.75, 2.25)],
                               facecolor=PALE_BLUE, edgecolor=BLUE, lw=1.5))
    main.plot([2.35, 2.35], [1.75, 2.25], color=ORANGE, lw=3)
    main.text(1.23, 2.75, "NTFF rear face\nin air", ha="right", color=GREEN, fontsize=9)
    main.text(2.35, 2.55, "port plane\nx = 12 mm", ha="center", color=ORANGE, fontsize=9)
    main.text(5.8, 2.1, "physical horn", ha="center", color="#164663", fontsize=11, weight="bold")
    main.text(9.43, 2.0, "PML", ha="center", rotation=90, fontsize=9)
    arrow(main, (2.55, 1.25), (5.4, 1.25), ORANGE, "launched wave", (0, 0.12))
    arrow(main, (5.4, 0.75), (2.55, 0.75), BLUE, "returning wave", (0, 0.12))
    main.text(0.35, 3.75, "Main simulation grid", fontsize=11, weight="bold")

    auxiliary.set(xlim=(0, 10), ylim=(0, 2.5))
    auxiliary.add_patch(Rectangle((1.2, 0.45), 7.3, 1.55, facecolor=PALE_BLUE, edgecolor=BLUE, lw=1.5))
    auxiliary.add_patch(Rectangle((1.2, 0.45), 0.65, 1.55, facecolor=PALE_GREY, edgecolor="none"))
    auxiliary.plot([8.5, 8.5], [0.45, 2.0], color=ORANGE, lw=3)
    auxiliary.text(1.5, 1.25, "PML", ha="center", fontsize=9)
    auxiliary.text(5.65, 1.25, "modal source", ha="center", fontsize=10)
    auxiliary.text(8.5, 2.15, "coupled aperture", ha="center", color=ORANGE, fontsize=9)
    arrow(auxiliary, (6.7, 0.85), (8.2, 0.85), ORANGE)
    arrow(auxiliary, (8.2, 0.6), (2.25, 0.6), BLUE)
    auxiliary.text(0.35, 2.2, "Separate virtual-waveguide grid", fontsize=11, weight="bold")
    for ax in (main, auxiliary):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    fig.savefig(OUTPUT / "virtual_horn_geometry.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    straight_guide()
    virtual_horn()


if __name__ == "__main__":
    main()
