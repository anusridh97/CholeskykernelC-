"""Make the headline figure: S-NIAH accuracy vs context length.

Overlays your SKA results (loaded from CSV) on top of PDN's published numbers
from their Table 5. Output: PDF + PNG suitable for the paper.

Usage:
  python plot_sniah.py --ours results/sniah_ckpt_final.csv --out figures/
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# PDN Table 5 (340M, 15B tokens). Replace with the exact numbers from the paper
# once you re-extract them; these are placeholders matching the figure shape.
PDN_NUMBERS = {
    # variant: { length: { model: accuracy } }
    1: {
        2048:  {"DN": 0.30, "GDN": 0.66, "PGDN": 0.71, "KDA": 0.68, "PKDA": 0.74},
        4096:  {"DN": 0.18, "GDN": 0.52, "PGDN": 0.62, "KDA": 0.55, "PKDA": 0.65},
        8192:  {"DN": 0.05, "GDN": 0.30, "PGDN": 0.40, "KDA": 0.32, "PKDA": 0.42},
        16384: {"DN": 0.02, "GDN": 0.12, "PGDN": 0.18, "KDA": 0.13, "PKDA": 0.20},
        32768: {"DN": 0.01, "GDN": 0.05, "PGDN": 0.08, "KDA": 0.06, "PKDA": 0.10},
    },
    2: {
        2048:  {"DN": 0.32, "GDN": 0.62, "PGDN": 0.68, "KDA": 0.65, "PKDA": 0.70},
        4096:  {"DN": 0.20, "GDN": 0.48, "PGDN": 0.56, "KDA": 0.50, "PKDA": 0.60},
        8192:  {"DN": 0.06, "GDN": 0.25, "PGDN": 0.35, "KDA": 0.27, "PKDA": 0.38},
        16384: {"DN": 0.02, "GDN": 0.10, "PGDN": 0.15, "KDA": 0.11, "PKDA": 0.18},
        32768: {"DN": 0.01, "GDN": 0.04, "PGDN": 0.06, "KDA": 0.04, "PKDA": 0.08},
    },
    3: {
        2048:  {"DN": 0.20, "GDN": 0.45, "PGDN": 0.52, "KDA": 0.48, "PKDA": 0.55},
        4096:  {"DN": 0.10, "GDN": 0.30, "PGDN": 0.40, "KDA": 0.33, "PKDA": 0.42},
        8192:  {"DN": 0.03, "GDN": 0.15, "PGDN": 0.22, "KDA": 0.17, "PKDA": 0.25},
        16384: {"DN": 0.01, "GDN": 0.06, "PGDN": 0.10, "KDA": 0.07, "PKDA": 0.12},
        32768: {"DN": 0.00, "GDN": 0.02, "PGDN": 0.04, "KDA": 0.02, "PKDA": 0.05},
    },
}


def load_ours(path: Path):
    out = defaultdict(dict)
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            v = int(row["variant"])
            L = int(row["context_length"])
            acc = float(row["accuracy"])
            out[v][L] = acc
    return out


def make_figure(ours, out_dir: Path, suffix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
    for i, variant in enumerate([1, 2, 3]):
        ax = axes[i]
        # PDN baselines.
        lengths = sorted(PDN_NUMBERS[variant].keys())
        for model, color in [
            ("DN", "#aaaaaa"),
            ("GDN", "#7da0ca"),
            ("PGDN", "#1f4e79"),
            ("KDA", "#e58a3a"),
            ("PKDA", "#a8431a"),
        ]:
            ys = [PDN_NUMBERS[variant][L][model] for L in lengths]
            ax.plot(lengths, ys, marker="o", lw=1.4, color=color, label=model)
        # Ours.
        if variant in ours:
            our_lengths = sorted(ours[variant].keys())
            our_ys = [ours[variant][L] for L in our_lengths]
            ax.plot(our_lengths, our_ys, marker="*", ms=11, lw=2.2,
                    color="#a30000", label="SKA (ours)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths)
        ax.set_xticklabels([f"{L//1024}K" for L in lengths])
        ax.set_xlabel("Context length")
        if i == 0:
            ax.set_ylabel("Accuracy")
        ax.set_title(f"S-NIAH-{variant}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        if i == 2:
            ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(
        "Long-context recall: SKA (exact) vs. PDN family (diagonal)",
        fontsize=12,
    )
    fig.tight_layout()
    pdf = out_dir / f"sniah_extrapolation{suffix}.pdf"
    png = out_dir / f"sniah_extrapolation{suffix}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"Wrote {pdf} and {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=str, required=True)
    ap.add_argument("--out", type=str, default="figures")
    ap.add_argument("--suffix", type=str, default="")
    args = ap.parse_args()

    ours = load_ours(Path(args.ours))
    make_figure(ours, Path(args.out), args.suffix)


if __name__ == "__main__":
    main()
