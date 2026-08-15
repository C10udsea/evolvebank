"""Generate the project's result figures from the experiment logs.

Figures:
  1. fig_evolution.png  -- learn-phase cumulative pass rate (top panel) and
     strategy-bank growth (bottom panel), sharing the x axis. Two panels,
     never a dual axis.
  2. fig_compare.png    -- frozen-bank vs no-bank pass rate on unseen test
     tasks, per seed dots + mean bar.

Style follows the dataviz reference palette (validated CVD-safe):
series-1 blue #2a78d6 (with bank), series-2 orange #eb6834 (without).
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- palette (reference instance, validated) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
S1 = "#2a78d6"  # with strategy bank
S2 = "#eb6834"  # without strategy bank

RESULTS = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")


def load(tag):
    path = os.path.join(RESULTS, f"selfevolve_{tag}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_evolution():
    rows = load("learn80")
    n = list(range(1, len(rows) + 1))
    cum = [r["cum_pass_rate"] for r in rows]
    # smooth the cumulative curve (window mean) to show the trend clearly
    w = 10
    smooth = [sum(cum[max(0, i - w + 1) : i + 1]) / len(cum[max(0, i - w + 1) : i + 1]) for i in range(len(cum))]
    bank = [r["bank_size"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]}
    )
    fig.patch.set_facecolor(SURFACE)

    style_ax(ax1)
    ax1.plot(n, cum, color=S1, linewidth=1.0, alpha=0.30, label="cumulative pass rate")
    ax1.plot(n, smooth, color=S1, linewidth=2.0, label="10-task moving average")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("pass rate", color=INK_2, fontsize=10)
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")

    style_ax(ax2)
    ax2.plot(n, bank, color=INK_2, linewidth=2.0)
    ax2.set_ylabel("bank size", color=INK_2, fontsize=10)
    ax2.set_xlabel("training tasks experienced", color=INK_2, fontsize=10)

    fig.suptitle("Self-evolution: agent pass rate while the strategy bank grows", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(RESULTS, "fig_evolution.png")
    fig.savefig(out, dpi=200)
    print("saved", out)


def fig_compare():
    groups = [
        ("with bank\n(frozen)", S1, ["eval_frozen_s10", "eval_frozen_s11"]),
        ("no bank\n(control)", S2, ["eval_none_s10", "eval_none_s11"]),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)

    stats = []
    for i, (label, color, tags) in enumerate(groups):
        rates = []
        for t in tags:
            try:
                rows = load(t)
            except FileNotFoundError:
                continue
            rates.append(sum(r["reward"] for r in rows) / len(rows))
        stats.append((label, color, rates))

    for i, (label, color, rates) in enumerate(stats):
        x = i
        mean = sum(rates) / len(rates)
        ax.bar([x], [mean], width=0.52, color=color, zorder=2)
        # per-seed dots next to the bar: show run-to-run variance honestly
        ax.scatter([x + 0.33] * len(rates), rates, s=34, color=color, zorder=3,
                   edgecolors=SURFACE, linewidths=1.2)
        for r in rates:
            ax.plot([x + 0.20, x + 0.46], [r, r], color=MUTED, linewidth=0.7, zorder=2)
        ax.text(x, mean + 0.02, f"{mean:.3f}", ha="center", fontsize=10, color=INK)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([s[0] for s in stats], fontsize=10, color=INK_2)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("pass rate on unseen test tasks", color=INK_2, fontsize=10)
    ax.set_title("tau-bench retail · 60 unseen tasks · 2 seeds (dots = per-seed)", color=INK, fontsize=10.5)

    out = os.path.join(RESULTS, "fig_compare.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("evolution", "all"):
        fig_evolution()
    if which in ("compare", "all"):
        fig_compare()
