"""
visualizer.py — Step 5 of the A2S pipeline.

Generates publication-quality figures from evaluator outputs.

Figures produced:
  1. Gap score bar chart (ΔAcc, ΔF1 per model)
  2. Level gradient line plot (Acc/MF1 A→B→C per model)
  3. Category heatmap (MF1 per category × model at each level)
  4. Baseline comparison grouped bar chart
  5. McNemar p-value table figure (all three pairs: A vs B, B vs C, A vs C)

Fixes applied:
  - FIX m3: Y-axis limits in fig_gap_scores are now computed dynamically from
    the actual data rather than being hardcoded to (-0.1, 0.6). Hardcoded
    limits silently clip bars if any gap exceeds 0.60.
  - FIX C3 follow-on: McNemar table now displays all three pairs (A vs B,
    B vs C, A vs C) rather than just A vs C, matching the updated evaluator.

Usage:
    python src/visualizer.py

Outputs:
    results/figures/*.png
    results/figures/*.pdf
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    MODELS, FIGURES, RESULTS, get_logger,
    load_json, timestamp,
)

log = get_logger("visualizer")

# ─── Style ────────────────────────────────────────────────────────────────────

PALETTE = {
    "gpt4o":    "#4E79A7",
    "claude":   "#F28E2B",
    "gemini":   "#59A14F",
    "deepseek": "#E15759",
    "level_A":  "#AEC6CF",
    "level_B":  "#FFB347",
    "level_C":  "#FF6961",
}

MODEL_LABELS = {
    "gpt4o":    "GPT-4o",
    "claude":   "Claude\nSonnet 4.6",
    "gemini":   "Gemini\n1.5 Pro",
    "deepseek": "DeepSeek\nV2.5",
}

LEVEL_LABELS = {
    "A": "Level A\n(Abstract)",
    "B": "Level B\n(Minimal)",
    "C": "Level C\n(Situated)",
}

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi":      150,
    "savefig.bbox":    "tight",
    "savefig.dpi":     300,
})


def safe_val(d: dict, *keys, default=float("nan")):
    """Safely navigate nested dicts."""
    v = d
    for k in keys:
        if not isinstance(v, dict):
            return default
        v = v.get(k, default)
    return v if v is not None else default


# ─── Figure 1: Gap Score Bar Chart ────────────────────────────────────────────

def fig_gap_scores(summary: dict) -> None:
    """
    FIX m3: Y-axis limits are now computed dynamically from the data.
    The previous hardcoded limit of 0.6 would silently clip bars if any
    model's gap exceeded that value (plausible given Cheung et al.'s 45pp
    omission bias finding). We now add a 10% margin above/below extremes.
    """
    models = [m for m in MODELS if m in summary]
    n      = len(models)
    x      = np.arange(n)
    width  = 0.35

    delta_acc = [safe_val(summary, m, "gap_score", "delta_acc") for m in models]
    delta_f1  = [safe_val(summary, m, "gap_score", "delta_f1")  for m in models]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width/2, delta_acc, width, label="ΔAccuracy",
                   color="#4E79A7", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, delta_f1,  width, label="ΔMacro-F1",
                   color="#F28E2B", edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Model")
    ax.set_ylabel("Performance Drop (Level A − Level C)")
    ax.set_title("Norm Grounding Gap by Model\n(positive = worse at Level C)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.legend(framealpha=0.9)

    # Dynamic y-axis limits with 10% margin
    all_vals = [v for v in delta_acc + delta_f1 if not np.isnan(v)]
    if all_vals:
        lo = min(all_vals)
        hi = max(all_vals)
        margin = max(0.05, (hi - lo) * 0.15)
        ax.set_ylim(min(-0.05, lo - margin), hi + margin)
    else:
        ax.set_ylim(-0.1, 0.6)  # sensible default if no data

    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}")
    )

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            va = "bottom" if h >= 0 else "top"
            offset = 0.005 if h >= 0 else -0.005
            ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                    f"{h:.3f}", ha="center", va=va, fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            va = "bottom" if h >= 0 else "top"
            offset = 0.005 if h >= 0 else -0.005
            ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                    f"{h:.3f}", ha="center", va=va, fontsize=8)

    fig.tight_layout()
    _save(fig, "fig1_gap_scores")


# ─── Figure 2: Level Gradient Line Plot ───────────────────────────────────────

def fig_level_gradient(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax_idx, (metric, label) in enumerate([("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")]):
        ax = axes[ax_idx]
        for model in MODELS:
            if model not in summary:
                continue
            vals = [
                safe_val(summary, model, f"level_{L}", metric)
                for L in ["A", "B", "C"]
            ]
            ax.plot(["A", "B", "C"], vals,
                    marker="o", linewidth=2,
                    color=PALETTE.get(model, "gray"),
                    label=MODEL_LABELS.get(model, model))
            # Annotate A→C gap with a dotted double-headed arrow
            vA, vC = vals[0], vals[2]
            if not (np.isnan(vA) or np.isnan(vC)):
                ax.annotate("", xy=(2, vC), xytext=(0, vA),
                            arrowprops=dict(
                                arrowstyle="<->",
                                color=PALETTE.get(model, "gray"),
                                lw=1, linestyle="dotted",
                            ))

        ax.set_xlabel("Abstraction Level")
        ax.set_ylabel(label)
        ax.set_title(f"{label}: Abstract → Situated Gradient")
        ax.set_ylim(0.2, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower left", framealpha=0.9)

    fig.suptitle("A2S Level Gradient — Performance Across Abstraction Levels",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_level_gradient")


# ─── Figure 3: Category Heatmap ───────────────────────────────────────────────

def fig_category_heatmap(cat_breakdown: dict) -> None:
    models = [m for m in MODELS if m in cat_breakdown]
    if not models:
        log.warning("No category breakdown data — skipping heatmap.")
        return

    for level in ["A", "B", "C"]:
        data_matrix = []
        for model in models:
            row = [
                safe_val(cat_breakdown, model, f"level_{level}", cat, "macro_f1")
                for cat in CATEGORIES
            ]
            data_matrix.append(row)

        arr = np.array(data_matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        im = ax.imshow(arr, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")

        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_xticklabels([c.capitalize() for c in CATEGORIES])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models])

        for i in range(len(models)):
            for j in range(len(CATEGORIES)):
                v   = arr[i, j]
                txt = f"{v:.2f}" if not np.isnan(v) else "—"
                col = "white" if v < 0.4 or v > 0.75 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=col)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Macro-F1")
        ax.set_title(
            f"Category-Level Macro-F1 — Level {level} "
            f"({LEVEL_LABELS[level].replace(chr(10), ' ')})"
        )
        fig.tight_layout()
        _save(fig, f"fig3_category_heatmap_level{level}")


# ─── Figure 4: Baseline Comparison ────────────────────────────────────────────

def fig_baseline_comparison(summary: dict, baselines: dict) -> None:
    models = [m for m in MODELS if m in summary]
    if not models:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax_idx, (metric, label) in enumerate([("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")]):
        ax = axes[ax_idx]
        x  = np.arange(len(models))

        for lvl_idx, level in enumerate(["A", "C"]):
            vals   = [safe_val(summary, m, f"level_{level}", metric) for m in models]
            offset = (lvl_idx - 0.5) * 0.25
            color  = PALETTE.get(f"level_{level}", "gray")
            ax.bar(x + offset, vals, 0.25,
                   label=f"Level {level}", color=color,
                   edgecolor="white", linewidth=0.5)

        style_map = {
            "always_violation":     ("--", "#E15759", "Always-Violation"),
            "always_non_violation": (":",  "#59A14F", "Always-Non-Violation"),
            "random":               ("-.", "gray",    "Random"),
        }
        for bname, (ls, bc, blabel) in style_map.items():
            bval = safe_val(baselines, bname, metric)
            ax.axhline(bval, linestyle=ls, color=bc, linewidth=1.5,
                       label=f"{blabel} ({bval:.2f})")

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{label} vs. Baselines")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Model Performance vs. Baselines at Level A and Level C",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_baseline_comparison")


# ─── Figure 5: McNemar Summary Table ─────────────────────────────────────────

def fig_mcnemar_table(mcnemar: dict) -> None:
    """
    FIX C3 follow-on: The table now displays all three comparison pairs
    (A vs B, B vs C, A vs C) per model rather than only A vs C.
    This reflects the updated evaluator which runs McNemar for all pairs.
    """
    models = [m for m in MODELS if m in mcnemar]
    if not models:
        log.warning("No McNemar data — skipping table figure.")
        return

    PAIRS = ["A_vs_B", "B_vs_C", "A_vs_C"]
    PAIR_LABELS = {"A_vs_B": "A vs B", "B_vs_C": "B vs C", "A_vs_C": "A vs C"}

    headers = ["Model", "Pair", "b (L1✓,L2✗)", "c (L1✗,L2✓)", "χ²", "p-value", "Sig."]
    rows = []
    for m in models:
        model_label = MODEL_LABELS.get(m, m).replace("\n", " ")
        for pair in PAIRS:
            mc = mcnemar[m].get(pair)
            if mc is None:
                rows.append([model_label, PAIR_LABELS[pair], "—", "—", "—", "—", "—"])
                continue
            sig = "✓" if mc.get("significant") else "✗"
            rows.append([
                model_label,
                PAIR_LABELS[pair],
                str(mc.get("b", "—")),
                str(mc.get("c", "—")),
                f"{mc.get('chi2', float('nan')):.3f}",
                f"{mc.get('p_value', float('nan')):.4f}",
                sig,
            ])

    n_rows  = len(rows)
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.45 * n_rows))
    ax.axis("off")

    table = ax.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="center",
        colColours=["#DDEEFF"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    ax.set_title(
        "McNemar's Test: All Level Pairs (α = 0.05)\n"
        "b = correct at L1 but wrong at L2; c = wrong at L1 but correct at L2",
        fontsize=11, pad=14,
    )
    fig.tight_layout()
    _save(fig, "fig5_mcnemar_table")


# ─── Save helper ──────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    for ext in ["png", "pdf"]:
        out = FIGURES / f"{name}.{ext}"
        fig.savefig(out)
        log.info(f"  Saved: {out.name}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    metrics_path = RESULTS / "metrics_summary.json"
    mcnemar_path = RESULTS / "mcnemar_tests.json"
    cat_path     = RESULTS / "category_breakdown.json"

    if not metrics_path.exists():
        log.error("metrics_summary.json not found. Run evaluator.py first.")
        return

    data      = load_json(metrics_path)
    summary   = data.get("models", {})
    baselines = data.get("baselines", {})

    mcnemar_data = load_json(mcnemar_path) if mcnemar_path.exists() else {}
    cat_data     = load_json(cat_path)     if cat_path.exists()    else {}

    log.info("Generating figures...")
    fig_gap_scores(summary)
    fig_level_gradient(summary)
    fig_category_heatmap(cat_data)
    fig_baseline_comparison(summary, baselines)
    fig_mcnemar_table(mcnemar_data)

    log.info(f"\nAll figures saved to {FIGURES}/")


if __name__ == "__main__":
    main()
