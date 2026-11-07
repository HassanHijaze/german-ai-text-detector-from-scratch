"""Draw the six comparison figures and save them to results/ as 300-dpi PNG.

Uses the statistics computed by analyze.py (run directly or via run.py).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze
from common import COLORS, FIG_DIR, GEN_NAMES, LENGTH_LABELS, SRC_NAMES, length_bin

GREEN, YELLOW, RED, GREY = "#4C9F70", "#F2B134", "#D1495B", "#BBBBBB"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "xtick.color": "#444444", "ytick.color": "#444444",
    "legend.frameon": False, "figure.dpi": 110,
    "savefig.bbox": "tight", "svg.fonttype": "none",
})
PERCENT = mpl.ticker.PercentFormatter(1.0, decimals=0)


# ---------- Helpers ----------
def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  {name}.png")


def set_title(ax, title, subtitle):
    ax.set_title(title, pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#555555", va="bottom")


def text_color(value, vmin, vmax):
    return "white" if (value - vmin) / (vmax - vmin + 1e-12) > 0.55 else "#222222"


def clean_heatmap_axes(ax, im, fig, cbar_label, cbar_format=None):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, format=cbar_format)
    cbar.set_label(cbar_label)
    cbar.outline.set_visible(False)


def triangle_heatmap(names, values, cell_text, cmap, vmin, vmax, cbar_label):
    """Lower-triangle heatmap of a symmetric model × model matrix (no diagonal)."""
    n = len(names)
    data = np.full((n - 1, n - 1), np.nan)
    for i in range(n - 1):
        for j in range(i + 1):
            data[i, j] = values[i + 1][j]

    fig, ax = plt.subplots(figsize=(7, 5.6))
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(n - 1):
        for j in range(i + 1):
            ax.text(j, i, cell_text[i + 1][j], ha="center", va="center", fontsize=9,
                    color=text_color(data[i, j], vmin, vmax))
    ax.set_xticks(range(n - 1), names[:-1], rotation=30, ha="right")
    ax.set_yticks(range(n - 1), names[1:])
    clean_heatmap_axes(ax, im, fig, cbar_label)
    return fig, ax


def group_heatmap(table, names, group_names):
    """Models (rows) × groups (columns) heatmap of error rates."""
    groups = list(table.index)
    data = table[names].T.values
    vmax = data.max()

    fig, ax = plt.subplots(figsize=(1.9 * len(groups) + 3, 4.6))
    im = ax.imshow(data, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
    for i in range(len(names)):
        for j in range(len(groups)):
            ax.text(j, i, f"{data[i, j]:.1%}", ha="center", va="center", fontsize=10,
                    color=text_color(data[i, j], 0, vmax))
    ax.set_xticks(range(len(groups)),
                  [f"{group_names.get(g, g)}\n(n = {int(table.loc[g, 'samples'])})" for g in groups])
    ax.set_yticks(range(len(names)), names)
    clean_heatmap_axes(ax, im, fig, "Error rate", cbar_format=PERCENT)
    return fig, ax


# ---------- Figures ----------
def plot_mcnemar(names, mcnemar):
    n = len(names)
    strength = [[np.nan] * n for _ in range(n)]
    labels = [[""] * n for _ in range(n)]
    for _, r in mcnemar.iterrows():
        i, j = names.index(r.model_a), names.index(r.model_b)
        row, col = max(i, j), min(i, j)
        net = r.a_right_b_wrong - r.a_wrong_b_right            # model_a over model_b
        net_row = net if names[row] == r.model_a else -net     # row model over column model
        p = r.p_holm
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        strength[row][col] = min(-np.log10(max(p, 1e-300)), 10)
        labels[row][col] = f"{p_text} {stars}\nnet {net_row:+d}"

    fig, ax = triangle_heatmap(names, strength, labels, "Blues", 0, 10, "−log₁₀(p)  (capped at 10)")
    set_title(ax, "Pairwise McNemar tests",
              "Holm-corrected. 'net' = texts the row model gets right minus texts the column model gets right.")
    save(fig, "mcnemar-heatmap")


def plot_agreement_summary(summary):
    s = summary.iloc[0]
    parts = [("All models correct", s.all_models_correct, GREEN),
             ("Models disagree", s.models_disagree, YELLOW),
             ("All models wrong", s.all_models_wrong, RED)]

    fig, ax = plt.subplots(figsize=(10, 2.2))
    left, handles = 0.0, []
    for label, count, color in parts:
        share = count / s.texts
        handles.append(ax.barh(0, share, left=left, color=color, height=0.5,
                               label=f"{label}: {int(count):,} ({share:.1%})"))
        if share >= 0.15:  # label inside the bar only when it fits
            ax.text(left + share / 2, 0, f"{share:.1%}", ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")
        left += share

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=3, fontsize=9, handlelength=1.2, columnspacing=2)
    set_title(ax, "How often do the five models agree?", f"All {int(s.texts):,} test texts.")
    save(fig, "agreement-summary")


def plot_error_by_generator(names, table):
    fig, ax = group_heatmap(table, names, GEN_NAMES)
    set_title(ax, "Error rate by generator",
              "AI-generated and AI-assisted test texts only. Rows: models, columns: the LLM that produced the text.")
    save(fig, "error-by-generator")


def plot_error_by_source(names, table):
    fig, ax = group_heatmap(table, names, SRC_NAMES)
    set_title(ax, "Error rate by source",
              "All test texts. Note: DeepSeek texts come mostly from peDOCS, so source and generator overlap.")
    save(fig, "error-by-source")


def plot_confidence(names, confidence):
    ct = confidence.set_index("model").loc[names]
    ypos = np.arange(len(names))[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})

    # Left: mean confidence when right vs wrong
    for yy, name in zip(ypos, names):
        right, wrong = ct.loc[name, "mean_conf_correct"], ct.loc[name, "mean_conf_wrong"]
        ax1.plot([wrong, right], [yy, yy], color=GREY, lw=2.5, zorder=1)
        ax1.scatter(right, yy, s=70, color=GREEN, zorder=2, label="Correct" if yy == ypos[0] else None)
        ax1.scatter(wrong, yy, s=70, color=RED, zorder=2, label="Wrong" if yy == ypos[0] else None)
    ax1.set_yticks(ypos, names)
    ax1.set_xlim(0.85, 1.005)
    ax1.xaxis.set_major_formatter(PERCENT)
    ax1.set_xlabel("Mean confidence of the predicted class")
    ax1.grid(axis="x", color="#EEEEEE")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.3, columnspacing=1.5)
    set_title(ax1, "Confidence when right vs wrong", "Wrong predictions are almost as confident as correct ones.")

    # Right: share of wrong predictions with confidence >= 90%
    share = ct["wrong_with_conf_over_90"].values
    ax2.barh(ypos, share, color=[COLORS[n] for n in names], height=0.6)
    for yy, v in zip(ypos, share):
        ax2.text(v + 0.01, yy, f"{v:.0%}", va="center", fontsize=9)
    ax2.set_yticks(ypos, [""] * len(names))
    ax2.set_xlim(0, 1.08)
    ax2.xaxis.set_major_formatter(PERCENT)
    ax2.set_xlabel("Share of wrong predictions")
    set_title(ax2, "Wrong but ≥ 90% confident", "High confidence does not mean the prediction is right.")

    fig.tight_layout()
    save(fig, "confidence-correct-vs-wrong")


def plot_hardest_by_length(meta, correct):
    all_wrong = (~correct).all(axis=1)
    bins = length_bin(meta["word_count"])
    whole = bins.value_counts(normalize=True).reindex(LENGTH_LABELS)
    hard = bins[all_wrong].value_counts(normalize=True).reindex(LENGTH_LABELS)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(LENGTH_LABELS))
    ax.bar(x - 0.2, whole.values, 0.4, color=GREY, label=f"Whole test set (n = {len(meta):,})")
    ax.bar(x + 0.2, hard.values, 0.4, color=RED, label=f"Wrong in all models (n = {int(all_wrong.sum())})")
    for xi, v in zip(x, whole.values):
        ax.text(xi - 0.2, v + 0.01, f"{v:.0%}", ha="center", fontsize=8, color="#555555")
    for xi, v in zip(x, hard.values):
        ax.text(xi + 0.2, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x, LENGTH_LABELS)
    ax.set_xlabel("Text length (words)")
    ax.yaxis.set_major_formatter(PERCENT)
    ax.set_ylabel("Share of texts")
    ax.legend(loc="upper right")
    set_title(ax, "Texts that fool every model are often short",
              "Length distribution of the texts all models misclassify, compared with the whole test set.")
    save(fig, "hardest-texts-by-length")


def main(results=None):
    if results is None:
        results = analyze.main()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    names = results["names"]

    print(f"Saving figures to {FIG_DIR}")
    plot_mcnemar(names, results["mcnemar"])
    plot_agreement_summary(results["summary"])
    plot_error_by_generator(names, results["by_generator"])
    plot_error_by_source(names, results["by_source"])
    plot_confidence(names, results["confidence"])
    plot_hardest_by_length(results["meta"], results["correct"])


if __name__ == "__main__":
    main()