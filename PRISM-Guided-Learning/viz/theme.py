"""Shared figure styling: the reference palette's surface and ink colours, and the axes style."""
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# Ablation families (configs/plot/ablation_summary.yaml), the same colour in every ablation figure
FAMILY_COLORS = {"Baselines": "#2a78d6", "Retry sweep": "#eb6834", "Feedback content": "#1baf7a",
                 "Blame signal": "#eda100", "Rounds budget": "#e87ba4", "Legacy variants": "#4a3aa7",
                 "Model": "#008300", "Rule vocabulary": "#e34948"}


def style(ax, title, ylabel=None, pad=10):
    """Bold left-aligned title, light horizontal grid, only the bottom spine. `pad=None`: matplotlib's default."""
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=pad)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
