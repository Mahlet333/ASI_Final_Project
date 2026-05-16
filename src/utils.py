"""
utils.py — Shared utilities for the A2S Transfer Task pipeline.
"""

import json
import os
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import jsonlines
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent

load_dotenv(dotenv_path=ROOT / ".env", override=False)

# ─── Logging ─────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_RAW         = ROOT / "data" / "raw"
DATA_CONSTRUCTED = ROOT / "data" / "constructed"
DATA_API_OUTPUTS = ROOT / "data" / "api_outputs"
DATA_PARSED      = ROOT / "data" / "parsed"
RESULTS          = ROOT / "results"
FIGURES          = RESULTS / "figures"

for p in [DATA_RAW, DATA_CONSTRUCTED, DATA_API_OUTPUTS, DATA_PARSED, RESULTS, FIGURES]:
    p.mkdir(parents=True, exist_ok=True)


# ─── Schema ──────────────────────────────────────────────────────────────────

CATEGORIES = ["politeness", "fairness", "honesty", "authority"]
LEVELS     = ["A", "B", "C"]

# ── OpenRouter config ────────────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")

MODELS = {
    "gpt4o":    os.getenv("OPENROUTER_GPT4O_MODEL",    "openai/gpt-4o"),
    "claude":   os.getenv("OPENROUTER_CLAUDE_MODEL",   "anthropic/claude-sonnet-4-5"),
    "gemini":   os.getenv("OPENROUTER_GEMINI_MODEL",   "google/gemini-2.5-flash"),
    "deepseek": os.getenv("OPENROUTER_DEEPSEEK_MODEL", "deepseek/deepseek-chat"),
}

VIOLATION_RATIO = 2   # violations : foils
N_VIOLATIONS    = 40
N_FOILS         = 20
N_TOTAL         = N_VIOLATIONS + N_FOILS   # 60
N_PER_CATEGORY  = 10  # violations per category
N_FOILS_PER_CAT =  5  # foils per category
N_RUNS          = int(os.getenv("N_RUNS", 3))
TEMPERATURE     = float(os.getenv("TEMPERATURE", 0))

# FIX C1: Paper Section 3.1 explicitly states κ ≥ 0.80 to select only norms
# that human raters found unambiguously violating. The previous value of 0.70
# was the lower bound for "acceptable" agreement, not high-confidence agreement.
KAPPA_THRESHOLD = 0.80


# ─── Item schema helpers ──────────────────────────────────────────────────────

def make_item_id(category: str, idx: int, is_violation: bool) -> str:
    kind = "V" if is_violation else "F"
    return f"{category[:3].upper()}-{kind}-{idx:03d}"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with jsonlines.open(path) as reader:
        return list(reader)


def save_jsonl(records: list[dict], path: Path, mode: str = "w") -> None:
    with jsonlines.open(path, mode=mode) as writer:
        writer.write_all(records)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict | list, path: Path) -> None:
    def _default(o):
        if isinstance(o, (bool, int, float, str)):
            return o
        return str(o)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─── JSON parsing with fallback ───────────────────────────────────────────────

def parse_violation_response(raw: str) -> Optional[bool]:
    """
    Try to extract {"violation": true/false} from model output.
    Falls back to targeted keyword search if JSON parsing fails.
    Returns None if both approaches fail (item marked null).

    FIX m1: The previous broad fallback matched any bare 'true'/'false'
    anywhere in the response, which could produce false parses (e.g.
    "True social norms are complex" → True). The fallback now requires
    proximity to the 'violation' key to avoid misclassification.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Attempt JSON parse
    try:
        obj = json.loads(cleaned)
        if "violation" in obj:
            return bool(obj["violation"])
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: require JSON-like key:value pattern
    lower = raw.lower()
    if re.search(r'"violation"\s*:\s*true', lower):
        return True
    if re.search(r'"violation"\s*:\s*false', lower):
        return False

    # Last resort: 'true'/'false' must appear within 30 chars of 'violation'
    m_true  = re.search(r'\btrue\b',      lower)
    m_false = re.search(r'\bfalse\b',     lower)
    m_viol  = re.search(r'\bviolation\b', lower)
    if m_viol and m_true  and abs(m_viol.start() - m_true.start())  < 30:
        return True
    if m_viol and m_false and abs(m_viol.start() - m_false.start()) < 30:
        return False

    return None   # parse failure


def parse_explanation_response(raw: str) -> Optional[str]:
    """Extract explanation string from Call 2 response."""
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        obj = json.loads(cleaned)
        return obj.get("explanation", None)
    except (json.JSONDecodeError, TypeError):
        return raw.strip() if raw.strip() else None


# ─── Prompt templates (frozen before inference) ───────────────────────────────

PROMPT_SYSTEM = (
    "You are evaluating social behaviour. "
    "Respond only with valid JSON. No preamble, no explanation."
)

def prompt_level_A(behavior: str) -> str:
    return (
        f"Description: {behavior}\n\n"
        'Is this a social norm violation? Respond with JSON only: {"violation": true or false}'
    )

def prompt_level_B(situation: str) -> str:
    return (
        f"Situation: {situation}\n\n"
        'Does this contain a social norm violation? Respond with JSON only: {"violation": true or false}'
    )

def prompt_level_C_call1(dialogue: list[dict]) -> str:
    turns = "\n".join(f"[{t['speaker']}]: {t['text']}" for t in dialogue)
    return (
        f"You are reading a conversation:\n\n{turns}\n\n"
        'Does this conversation contain a social norm violation? '
        'Respond with JSON only: {"violation": true or false}'
    )

def prompt_level_C_call2(dialogue: list[dict], judgment: bool) -> str:
    turns = "\n".join(f"[{t['speaker']}]: {t['text']}" for t in dialogue)
    verdict = "did" if judgment else "did not"
    return (
        f"You previously judged that this conversation {verdict} contain a social norm violation.\n\n"
        f"Conversation:\n{turns}\n\n"
        'In one sentence, explain your judgment. '
        'Respond with JSON only: {"explanation": "..."}'
    )


# ─── Timestamp ───────────────────────────────────────────────────────────────

def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
