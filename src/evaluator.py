"""
evaluator.py — Final simplified evaluator for the manual A2S pipeline.

Pipeline:
    Manual JSONL dataset → inference → evaluation → visualization

Computes:
  - Accuracy
  - Macro-F1
  - Per-class F1
  - Precision / Recall
  - Norm Grounding Gap Score

Usage:
    python src/evaluator.py

Outputs:
    results/metrics_summary.json
    results/per_item_results.jsonl
"""

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    MODELS,
    DATA_PARSED,
    RESULTS,
    get_logger,
    load_jsonl,
    save_json,
    save_jsonl,
    timestamp,
)

log = get_logger("evaluator")


def compute_metrics(y_true: list[bool],
                    y_pred: list[Optional[bool]]) -> dict:

    paired = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]

    n_null = sum(1 for p in y_pred if p is None)

    if not paired:
        return {
            "error": "all_null",
            "n_null": n_null,
        }

    yt = [t for t, _ in paired]
    yp = [p for _, p in paired]

    n = len(paired)

    TP = sum(1 for t, p in zip(yt, yp) if t and p)
    TN = sum(1 for t, p in zip(yt, yp) if not t and not p)
    FP = sum(1 for t, p in zip(yt, yp) if not t and p)
    FN = sum(1 for t, p in zip(yt, yp) if t and not p)

    acc = (TP + TN) / n

    prec_v = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec_v = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    f1_v = (
        2 * prec_v * rec_v / (prec_v + rec_v)
        if (prec_v + rec_v) > 0 else 0.0
    )

    prec_nv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    rec_nv = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    f1_nv = (
        2 * prec_nv * rec_nv / (prec_nv + rec_nv)
        if (prec_nv + rec_nv) > 0 else 0.0
    )

    macro_f1 = (f1_v + f1_nv) / 2

    return {
        "n_items": n,
        "n_null": n_null,

        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),

        "f1_violation": round(f1_v, 4),
        "f1_non_violation": round(f1_nv, 4),

        "precision_violation": round(prec_v, 4),
        "recall_violation": round(rec_v, 4),

        "precision_non_violation": round(prec_nv, 4),
        "recall_non_violation": round(rec_nv, 4),

        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }


def compute_gap(metrics_A: dict,
                metrics_C: dict) -> dict:

    if "error" in metrics_A or "error" in metrics_C:
        return {"error": "cannot_compute_gap"}

    delta_acc = round(
        metrics_A["accuracy"] - metrics_C["accuracy"],
        4
    )

    delta_f1 = round(
        metrics_A["macro_f1"] - metrics_C["macro_f1"],
        4
    )

    return {
        "delta_acc": delta_acc,
        "delta_f1": delta_f1,
    }


def preflight_check(models: list[str]) -> None:

    missing = []

    for model_key in models:

        for level in ["A", "B", "C"]:

            path = DATA_PARSED / f"{model_key}_level{level}.jsonl"

            if not path.exists():
                missing.append(str(path))

    if missing:

        log.warning("Missing parsed files:")

        for m in missing:
            log.warning(f"  {m}")


def load_parsed(model_key: str,
                level: str) -> list[dict]:

    path = DATA_PARSED / f"{model_key}_level{level}.jsonl"

    return load_jsonl(path)


def main() -> None:

    preflight_check(list(MODELS.keys()))

    summary = {}

    all_per_item = []

    for model_key in MODELS:

        log.info(f"=== {model_key} ===")

        summary[model_key] = {}

        level_data = {}

        for level in ["A", "B", "C"]:

            records = load_parsed(model_key, level)

            if not records:
                continue

            y_true = [r["is_violation"] for r in records]

            y_pred = [r.get("judgment_final") for r in records]

            metrics = compute_metrics(y_true, y_pred)

            summary[model_key][f"level_{level}"] = metrics

            level_data[level] = {
                "records": records,
                "y_true": y_true,
                "y_pred": y_pred,
            }

            log.info(
                f"Level {level}: "
                f"Acc={metrics.get('accuracy', '?')}  "
                f"MF1={metrics.get('macro_f1', '?')}"
            )

        if "A" in level_data and "C" in level_data:

            gap = compute_gap(
                summary[model_key]["level_A"],
                summary[model_key]["level_C"]
            )

            summary[model_key]["gap_score"] = gap

        for level in ["A", "B", "C"]:

            if level not in level_data:
                continue

            for r in level_data[level]["records"]:

                yt = r["is_violation"]

                yp = r.get("judgment_final")

                all_per_item.append({

                    "model": model_key,

                    "level": level,

                    "item_id": r["item_id"],

                    "is_violation": yt,

                    "judgment": yp,

                    "correct": (
                        yp == yt
                        if yp is not None else None
                    ),

                    "null": yp is None,
                })

    save_json(
        {
            "computed_at": timestamp(),
            "models": summary,
        },
        RESULTS / "metrics_summary.json"
    )

    if all_per_item:

        save_jsonl(
            all_per_item,
            RESULTS / "per_item_results.jsonl"
        )

    log.info(f"Results saved to {RESULTS}/")

    print_summary(summary)


def print_summary(summary: dict) -> None:

    print("\n" + "=" * 70)

    print("NORM GROUNDING GAP — RESULTS SUMMARY")

    print("=" * 70)

    print(
        f"{'Model':<12} "
        f"{'Lvl A':>8} "
        f"{'Lvl B':>8} "
        f"{'Lvl C':>8} "
        f"{'ΔAcc':>8} "
        f"{'ΔF1':>8}"
    )

    print("-" * 70)

    for model, data in summary.items():

        acc_A = data.get("level_A", {}).get("accuracy", float("nan"))

        acc_B = data.get("level_B", {}).get("accuracy", float("nan"))

        acc_C = data.get("level_C", {}).get("accuracy", float("nan"))

        gap = data.get("gap_score", {})

        d_acc = gap.get("delta_acc", float("nan"))

        d_f1 = gap.get("delta_f1", float("nan"))

        print(
            f"{model:<12} "
            f"{acc_A:>8.3f} "
            f"{acc_B:>8.3f} "
            f"{acc_C:>8.3f} "
            f"{d_acc:>8.3f} "
            f"{d_f1:>8.3f}"
        )

    print("-" * 70)

    print("=" * 70)


if __name__ == "__main__":
    main()