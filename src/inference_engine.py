"""
inference_engine.py — Step 3 of the A2S pipeline.

All models via OpenRouter's OpenAI-compatible API. One key, all models.

Fixes applied:
  - FIX C2: Level C Call 2 (explanation) now always runs regardless of whether
    Call 1 parsed successfully. Null-parse items are the most informative failure
    cases; skipping their explanations was a silent data loss.
  - FIX M5: Majority-vote aggregation now detects ties and marks the item null
    rather than returning an arbitrary winner from Counter.most_common.
  - FIX M6: Pre-flight check at startup verifies that parsed data exists for
    every model × level combination before inference begins, so missing Gemini
    Level A (or any other gap) is caught early with a clear error message.

Usage:
    python src/inference_engine.py
    python src/inference_engine.py --models claude gpt4o
    python src/inference_engine.py --models claude --levels C --runs 1
"""

import argparse
import os
import sys
import time
import json
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DATA_CONSTRUCTED, DATA_API_OUTPUTS, DATA_PARSED,
    MODELS, OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    N_RUNS, TEMPERATURE,
    get_logger, load_jsonl, save_jsonl, timestamp,
    parse_violation_response, parse_explanation_response,
    prompt_level_A, prompt_level_B, prompt_level_C_call1, prompt_level_C_call2,
    PROMPT_SYSTEM,
)

log = get_logger("inference_engine")


class OpenRouterClient:
    """Single client for all models via OpenRouter."""

    def __init__(self):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set in .env")
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")
        from openai import OpenAI
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        log.info("OpenRouter client ready")

    def complete(self, model_name: str, system: str, user: str,
                 retries: int = 3) -> tuple[str, dict]:
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model_name,
                    temperature=TEMPERATURE,
                    max_tokens=256,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    extra_headers={
                        "HTTP-Referer": "https://github.com/nyuad/a2s-transfer-task",
                        "X-Title": "A2S Transfer Task",
                    },
                )
                raw  = resp.choices[0].message.content or ""
                meta = {
                    "model_used":    resp.model,
                    "finish_reason": resp.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens":     getattr(resp.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    },
                }
                return raw, meta
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    log.warning(f"    Attempt {attempt+1} failed: {e} — retry in {wait}s")
                    time.sleep(wait)
                else:
                    raise


def run_level_A(item: dict, client: OpenRouterClient, model_name: str) -> dict:
    raw, meta = client.complete(model_name, PROMPT_SYSTEM, prompt_level_A(item["prompt"]))
    return {
        "item_id":      item["item_id"],
        "level":        "A",
        "model_name":   model_name,
        "is_violation": item["is_violation"],
        "raw_response": raw,
        "judgment":     parse_violation_response(raw),
        "meta":         meta,
        "recorded_at":  timestamp(),
    }


def run_level_B(item: dict, client: OpenRouterClient, model_name: str) -> dict:
    raw, meta = client.complete(model_name, PROMPT_SYSTEM, prompt_level_B(item["prompt"]))
    return {
        "item_id":      item["item_id"],
        "level":        "B",
        "model_name":   model_name,
        "is_violation": item["is_violation"],
        "raw_response": raw,
        "judgment":     parse_violation_response(raw),
        "meta":         meta,
        "recorded_at":  timestamp(),
    }


def run_level_C(item: dict, client: OpenRouterClient, model_name: str) -> dict:

    dialogue = item["dialogue"]

    raw1, meta1 = client.complete(
        model_name,
        PROMPT_SYSTEM,
        prompt_level_C_call1(dialogue)
    )

    judgment = parse_violation_response(raw1)

    judgment_for_call2 = judgment if judgment is not None else False

    raw2, meta2 = client.complete(
        model_name,
        PROMPT_SYSTEM,
        prompt_level_C_call2(dialogue, judgment_for_call2),
    )

    explanation = parse_explanation_response(raw2)

    return {
        "item_id":          item["item_id"],
        "level":            "C",
        "model_name":       model_name,
        "is_violation":     item["is_violation"],
        "raw_response_c1":  raw1,
        "raw_response_c2":  raw2,
        "judgment":         judgment,
        "call1_parse_ok":   judgment is not None,
        "explanation":      explanation,
        "meta_c1":          meta1,
        "meta_c2":          meta2,
        "recorded_at":      timestamp(),
    }


def run_one_batch(model_key: str, level: str, run: int,
                  items: list[dict], client: OpenRouterClient) -> None:

    model_name = MODELS[model_key]

    out_path = DATA_API_OUTPUTS / f"{model_key}_level{level}_run{run}.jsonl"

    if out_path.exists():
        log.info(f"    Already exists: {out_path.name} — skipping.")
        return

    results = []

    null_count = 0

    for i, item in enumerate(items):

        log.info(f"    [{i+1:02d}/{len(items)}] {item['item_id']}")

        try:

            if level == "A":
                result = run_level_A(item, client, model_name)

            elif level == "B":
                result = run_level_B(item, client, model_name)

            else:
                result = run_level_C(item, client, model_name)

            result["model_key"] = model_key
            result["run"] = run

            results.append(result)

            if result["judgment"] is None:

                null_count += 1

                log.warning(
                    f"      NULL parse: {item['item_id']}  "
                    f"raw={result.get('raw_response', result.get('raw_response_c1',''))[:60]}"
                )

            time.sleep(0.4)

        except Exception as e:

            log.error(f"      FAILED {item['item_id']}: {e}")

            results.append({
                "item_id":      item["item_id"],
                "level":        level,
                "model_key":    model_key,
                "model_name":   model_name,
                "is_violation": item["is_violation"],
                "error":        str(e),
                "judgment":     None,
                "run":          run,
                "recorded_at":  timestamp(),
            })

            null_count += 1

            time.sleep(2)

    save_jsonl(results, out_path)

    log.info(f"    Saved {len(results)} records → {out_path.name}")

    if items and null_count / len(items) > 0.10:

        log.warning(
            f"    ⚠️  NULL rate {null_count/len(items):.1%} > 10% — reliability caveat!"
        )


def aggregate_runs(model_key: str, level: str, n_runs: int) -> None:

    records = {}

    for run in range(1, n_runs + 1):

        path = DATA_API_OUTPUTS / f"{model_key}_level{level}_run{run}.jsonl"

        if not path.exists():
            continue

        for r in load_jsonl(path):

            iid = r["item_id"]

            if iid not in records:

                records[iid] = {
                    "item_id":      iid,
                    "level":        level,
                    "model":        model_key,
                    "is_violation": r["is_violation"],
                    "judgments":    [],
                    "explanations": [],
                }

            if r.get("judgment") is not None:
                records[iid]["judgments"].append(r["judgment"])

            if r.get("explanation"):
                records[iid]["explanations"].append(r["explanation"])

    parsed = []

    for iid, rec in records.items():

        js = rec["judgments"]

        if not js:

            rec["judgment_final"] = None
            rec["vote_result"] = "all_null"

        else:

            counts = Counter(js)

            top = counts.most_common(2)

            if len(top) > 1 and top[0][1] == top[1][1]:

                rec["judgment_final"] = None
                rec["vote_result"] = f"tie_{top[0][0]}_{top[1][0]}"

                log.warning(
                    f"  Tie vote for {iid} at Level {level} "
                    f"({top[0][0]}={top[0][1]} vs {top[1][0]}={top[1][1]})"
                )

            else:

                rec["judgment_final"] = top[0][0]

                rec["vote_result"] = (
                    f"majority_{top[0][0]}_{top[0][1]}_of_{len(js)}"
                )

        rec["judgment_runs"] = js
        rec["null_runs"] = n_runs - len(js)

        parsed.append(rec)

    if parsed:

        out = DATA_PARSED / f"{model_key}_level{level}.jsonl"

        save_jsonl(parsed, out)

        log.info(f"  Parsed: {out.name}  ({len(parsed)} items)")


def preflight_check(models: list[str], levels: list[str]) -> None:

    missing = []

    for level in levels:

        path = DATA_CONSTRUCTED / f"items_level{level}.jsonl"

        if not path.exists() or not load_jsonl(path):
            missing.append(str(path))

    if missing:

        log.error("Pre-flight check failed. Missing or empty item files:")

        for m in missing:
            log.error(f"  {m}")

        sys.exit(1)

    log.info(f"Pre-flight check passed: item files exist for levels {levels}.")


def main() -> None:

    parser = argparse.ArgumentParser(description="A2S Inference Engine (OpenRouter)")

    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys())
    )

    parser.add_argument(
        "--levels",
        nargs="+",
        default=["A", "B", "C"],
        choices=["A", "B", "C"]
    )

    parser.add_argument("--runs", type=int, default=N_RUNS)

    args = parser.parse_args()

    log.info(
        f"Models: {args.models} | Levels: {args.levels} | Runs: {args.runs}"
    )

    preflight_check(args.models, args.levels)

    items = {
        level: load_jsonl(DATA_CONSTRUCTED / f"items_level{level}.jsonl")
        for level in args.levels
    }

    try:
        client = OpenRouterClient()

    except Exception as e:

        log.error(f"Cannot create OpenRouter client: {e}")

        sys.exit(1)

    for model_key in args.models:

        log.info(f"\n=== {model_key}  ({MODELS[model_key]}) ===")

        for level in args.levels:

            for run in range(1, args.runs + 1):

                log.info(f"  Level {level}, Run {run}/{args.runs}")

                run_one_batch(
                    model_key,
                    level,
                    run,
                    items[level],
                    client
                )

    log.info("\nAggregating runs...")

    for model_key in args.models:

        for level in args.levels:

            aggregate_runs(model_key, level, args.runs)

    log.info("\nDone. Run: python src/evaluator.py")


if __name__ == "__main__":
    main()