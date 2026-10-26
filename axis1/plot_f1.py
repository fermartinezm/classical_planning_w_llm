"""Figure F1: test coverage per domain at L0-L3, grouped bars, h_FF as a marker.

Bars: mean coverage of the selected LLM heuristic over seeds, with the min-max range
across seeds as error bars. Black diamonds: Pyperplan h_FF on the same tasks at the same
level (should be flat across levels). One figure per model.

  python -m axis1.summarize      # writes axis1/results/summary.csv
  python -m axis1.plot_f1        # -> axis1/figures/f1_<model>.pdf and .png
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from axis1.config import DOMAINS, LEVELS, REPO_ROOT, RESULTS

COLORS = {"L0": "#2b6cb0", "L1": "#63a4d8", "L2": "#e8a33d", "L3": "#c0392b"}
LABELS = {"L0": "L0 original", "L1": "L1 renamed", "L2": "L2 renamed + shuffled", "L3": "L3 misleading"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=RESULTS / "summary.csv")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "axis1" / "figures")
    args = ap.parse_args(argv)

    rows = list(csv.DictReader(open(args.summary)))
    args.out.mkdir(parents=True, exist_ok=True)
    for model in sorted({r["model"] for r in rows}):
        levels = [l for l in LEVELS if any(r["model"] == model and r["obf_level"] == l for r in rows)]
        domains = [d for d in DOMAINS if any(r["model"] == model and r["domain"] == d for r in rows)]
        fig, ax = plt.subplots(figsize=(7.2, 2.8))
        width = 0.8 / len(levels)
        for i, level in enumerate(levels):
            xs, means, lo, hi, hff = [], [], [], [], []
            for j, d in enumerate(domains):
                r = next(r for r in rows if r["model"] == model and r["obf_level"] == level and r["domain"] == d)
                x = j - 0.4 + width * (i + 0.5)
                m = float(r["mean"])
                xs.append(x)
                means.append(m)
                lo.append(m - float(r["min"]))
                hi.append(float(r["max"]) - m)
                hff.append(float(r["hff"]) if r["hff"] not in ("", None) else None)
            ax.bar(xs, means, width, color=COLORS[level], label=LABELS[level],
                   yerr=[lo, hi] if any(lo) or any(hi) else None, capsize=2, error_kw={"lw": 0.8})
            hx = [x for x, h in zip(xs, hff) if h is not None]
            hy = [h for h in hff if h is not None]
            ax.scatter(hx, hy, marker="D", s=14, color="black", edgecolors="white", linewidths=0.6, zorder=3,
                       label="h_FF (same level)" if i == 0 and hx else None)
        n = int(rows[0]["n_tasks"])
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(f"tasks solved (of {n})", fontsize=8)
        ax.set_ylim(0, n + 1)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
        ax.set_title(f"Coverage vs. obfuscation level ({model})", fontsize=9, pad=22)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = args.out / f"f1_{model}.{ext}"
            fig.savefig(path, dpi=200)
        print("wrote", args.out / f"f1_{model}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
