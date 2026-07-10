"""Auto-generate figures from the experiment result JSON files.

Reads ``results/curvature_optimizer.json`` and ``results/gp_thompson.json``
(written by the two benchmarks) and renders publication-quality PNGs into
``figures/``, plus a single ``figures/report.html`` index. No GPU needed here —
this only consumes the JSON, so plots can be regenerated without re-running.

Colours follow the data-viz skill's validated categorical palette; identity is
carried by colour + direct labels (never colour alone), one y-axis per panel,
recessive grid/axes, thin marks.

Usage:
    python -m experiments.viz                 # render whatever JSON exists
    from experiments.viz import render_all    # called by run_experiments.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

# Validated categorical palette (light surface) from the data-viz skill.
BLUE, AQUA, YELLOW, GREEN, VIOLET, RED = (
    "#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
)
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e2"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.size": 11,
        "legend.frameon": False,
        "figure.dpi": 130,
    })


def _finish(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def _save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


# --------------------------------------------------------------------------- #
# Curvature optimizer figures
# --------------------------------------------------------------------------- #

_OPT_COLORS = {
    "AdamW": BLUE,
    "curvature (incremental)": AQUA,
    "curvature (periodic_refact K=20)": YELLOW,
}


def render_curvature(data):
    _style()
    figs = []
    training = data["training"]

    # 1. Loss curves.
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for name, res in training.items():
        color = _OPT_COLORS.get(name, VIOLET)
        curve = res["loss_curve"]
        ax.plot(range(1, len(curve) + 1), curve, color=color, lw=2, label=name)
        ax.annotate(name.replace("curvature ", ""), (len(curve), curve[-1]),
                    color=color, fontsize=9, va="center", ha="left",
                    xytext=(4, 0), textcoords="offset points")
    ax.axhline(data["target"], color=MUTED, lw=1, ls="--")
    ax.annotate("target", (1, data["target"]), color=MUTED, fontsize=8,
                va="bottom", ha="left")
    # Direct labels carry identity, so no legend box (avoids the collision).
    ax.set_xlim(right=len(curve) * 1.34)
    _finish(ax, "Training loss — incremental curvature optimizer vs AdamW",
            "step", "cross-entropy loss")
    figs.append(("curvature_loss.png", _save(fig, "curvature_loss.png")))

    # 2. Per-step optimizer cost.
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    names = list(training.keys())
    vals = [training[n]["med_opt_ms"] for n in names]
    colors = [_OPT_COLORS.get(n, VIOLET) for n in names]
    short = [n.replace("curvature ", "").replace(" K=20", "") for n in names]
    bars = ax.bar(short, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9, color=INK)
    _finish(ax, "Optimizer cost per step (backward excluded)",
            "", "median ms / step")
    figs.append(("curvature_stepcost.png", _save(fig, "curvature_stepcost.png")))

    # 3. Batched-vs-sequential refresh scaling (GPU only).
    rs = data.get("refresh_scaling", {})
    if rs.get("available"):
        rows = rs["rows"]
        T = [r["T"] for r in rows]
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.plot(T, [r["seq_ms"] for r in rows], color=RED, lw=2, marker="o",
                ms=5, label="sequential (one block at a time)")
        ax.plot(T, [r["bat_ms"] for r in rows], color=AQUA, lw=2, marker="o",
                ms=5, label="batched (one launch)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        _finish(ax, f"Chart refresh across a net's same-rank blocks (r={rs['r']})",
                "number of blocks (T)", "ms per refresh")
        ax.legend(loc="upper left")
        figs.append(("curvature_refresh_scaling.png",
                     _save(fig, "curvature_refresh_scaling.png")))
    return figs


# --------------------------------------------------------------------------- #
# GP Thompson figures
# --------------------------------------------------------------------------- #


def render_gp(data):
    _style()
    figs = []
    scaling = data["scaling"]
    T = [s["T"] for s in scaling]

    # 1. ms/round vs T for the three paths.
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(T, [s["batched_ms"] for s in scaling], color=AQUA, lw=2, marker="o",
            ms=5, label="batched (our kernel, one launch)")
    seqT = [s["T"] for s in scaling if s["seq_ms"]]
    if seqT:
        ax.plot(seqT, [s["seq_ms"] for s in scaling if s["seq_ms"]], color=BLUE,
                lw=2, marker="o", ms=5, label="sequential (our kernel, per arm)")
    boT = [s["T"] for s in scaling if s["botorch_ms"]]
    if boT:
        label = f"{scaling[0].get('botorch_backend') or 'botorch'} (per-model, cubic)"
        ax.plot(boT, [s["botorch_ms"] for s in scaling if s["botorch_ms"]],
                color=RED, lw=2, marker="o", ms=5, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _finish(ax, "Per-round conditioning cost vs number of GP arms",
            "arms (T)", "ms / round")
    ax.legend(loc="upper left")
    figs.append(("gp_cost_vs_T.png", _save(fig, "gp_cost_vs_T.png")))

    # 2. Throughput: arms conditioned per second.
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(T, [s["arms_per_sec"] for s in scaling], color=AQUA, lw=2, marker="o", ms=5)
    ax.fill_between(T, [s["arms_per_sec"] for s in scaling], color=AQUA, alpha=0.10)
    ax.set_xscale("log", base=2)
    for fmt_ax in (ax.xaxis,):
        fmt_ax.set_major_formatter(ScalarFormatter())
    _finish(ax, "Massively parallel throughput — arms conditioned per second",
            "arms (T)", "arms / second")
    figs.append(("gp_throughput.png", _save(fig, "gp_throughput.png")))

    # 3. Regret curve.
    rc = data.get("regret_curve")
    if rc:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        vals = rc["values"]
        ax.plot(range(1, len(vals) + 1), vals, color=VIOLET, lw=2, marker="o", ms=4)
        ax.fill_between(range(1, len(vals) + 1), vals, color=VIOLET, alpha=0.10)
        _finish(ax, f"Mean simple regret over rounds (T={rc['T']} arms)",
                "round", "mean simple regret")
        figs.append(("gp_regret.png", _save(fig, "gp_regret.png")))
    return figs


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #


def _report_html(sections):
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;
    margin:2rem auto;padding:0 1rem;color:#0b0b0b;background:#fcfcfb}
    h1{font-size:1.6rem} h2{font-size:1.2rem;margin-top:2.2rem;
    border-bottom:1px solid #e6e5e2;padding-bottom:.3rem}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}
    figure{margin:0} img{width:100%;border:1px solid #e6e5e2;border-radius:8px}
    figcaption{color:#52514e;font-size:.85rem;margin-top:.3rem}
    .meta{color:#52514e;font-size:.9rem}
    """
    parts = [f"<!doctype html><meta charset='utf-8'><title>Kernel experiments</title>",
             f"<style>{css}</style>",
             "<h1>Double-sided Cholesky kernel — GPU experiments</h1>"]
    for title, meta, figs in sections:
        parts.append(f"<h2>{title}</h2>")
        if meta:
            parts.append(f"<p class='meta'>{meta}</p>")
        parts.append("<div class='grid'>")
        for _, fname in figs:
            parts.append(f"<figure><img src='{fname}' alt='{fname}'>"
                         f"<figcaption>{fname}</figcaption></figure>")
        parts.append("</div>")
    return "\n".join(parts)


def render_all():
    FIG_DIR.mkdir(exist_ok=True)
    sections = []
    made = []

    cpath = RESULTS_DIR / "curvature_optimizer.json"
    if cpath.exists():
        data = json.loads(cpath.read_text())
        figs = render_curvature(data)
        made += figs
        meta = (f"device: {data['device']}"
                + (f" ({data['gpu']})" if data.get("gpu") else "")
                + f" · {data['steps']} steps · rank {data['rank']} · {data['timestamp']}")
        sections.append(("Incremental curvature optimizer", meta, figs))

    gpath = RESULTS_DIR / "gp_thompson.json"
    if gpath.exists():
        data = json.loads(gpath.read_text())
        figs = render_gp(data)
        made += figs
        meta = (f"device: {data['device']}"
                + (f" ({data['gpu']})" if data.get("gpu") else "")
                + f" · {data['candidates_r']} candidates · {data['rounds']} rounds"
                + f" · baseline: {data['botorch_backend']} · {data['timestamp']}")
        sections.append(("Massively parallel GP Thompson sampling", meta, figs))

    if not sections:
        print("No results found — run the benchmarks first.")
        return None
    html = FIG_DIR / "report.html"
    html.write_text(_report_html(sections))
    print(f"Rendered {len(made)} figures into {FIG_DIR}/")
    print(f"Report: {html}")
    return html


if __name__ == "__main__":
    render_all()
