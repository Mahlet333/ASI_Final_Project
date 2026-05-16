"""
test_endpoints.py — Run this BEFORE the full pipeline to verify all four
OpenRouter model endpoints are live and responding correctly.

Usage:
    python test_endpoints.py

Requires OPENROUTER_API_KEY in your .env file.
"""

import os, sys
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not API_KEY:
    print("ERROR: OPENROUTER_API_KEY not set in .env"); sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: run  conda install -c conda-forge openai"); sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

MODELS = {
    "gpt4o":    "openai/gpt-4o",
    "claude":   "anthropic/claude-sonnet-4-5",
    "gemini":   "google/gemini-2.5-flash",      # updated — gemini-pro-1.5 is dead
    "deepseek": "deepseek/deepseek-chat",
}

PROMPT = 'Is saying hello a social norm violation? JSON only: {"violation": true or false}'

print("\n" + "="*55)
print("  OpenRouter endpoint test")
print("="*55)

all_ok = True
for name, model_id in MODELS.items():
    try:
        resp = client.chat.completions.create(
            model=model_id,
            max_tokens=32,
            temperature=0,
            messages=[
                {"role": "system", "content": "Respond only with valid JSON."},
                {"role": "user",   "content": PROMPT},
            ],
        )
        answer = resp.choices[0].message.content.strip()
        actual = resp.model          # what OpenRouter actually served
        print(f"  ✓  {name:<10}  {model_id:<40}  →  {answer[:40]}")
        if actual != model_id:
            print(f"             (served as: {actual})")
    except Exception as e:
        print(f"  ✗  {name:<10}  {model_id:<40}  →  ERROR: {e}")
        all_ok = False

print("="*55)
if all_ok:
    print("  All endpoints OK — safe to run bash run_pipeline.sh\n")
else:
    print("  One or more endpoints FAILED — fix utils.py before running.\n")
