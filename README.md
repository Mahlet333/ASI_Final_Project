# A2S Transfer Task — Measuring the Norm Grounding Gap in LLMs

**CS-UH 3260 · Artificial Social Intelligence · NYU Abu Dhabi · Spring 2026**  
Authors: Aneeka Paul, Mahlet Astraw

---

## Overview

The Abstract-to-Situated (A2S) Transfer Task measures whether LLMs can detect social norm violations embedded in naturalistic four-turn conversations vs. isolated direct questions.

We define the **Norm Grounding Gap** as the drop in performance from Level A (abstract) to Level C (situated conversation).

---

## Project Structure

```
a2s_project/
├── README.md
├── requirements.txt
├── .env.example                  # API key template
│
├── data/
│   ├── raw/                      # NormBank source (place normbank.csv here)
│   ├── constructed/
│   │   ├── items_levelA.jsonl    # Generated after running dataset_constructor.py
│   │   ├── items_levelB.jsonl
│   │   ├── items_levelC.jsonl
│   │   └── dataset_manifest.json
│   ├── api_outputs/              # Raw API responses (never overwritten)
│   └── parsed/                   # Parsed binary judgments
│
├── src/
│   ├── dataset_constructor.py    # Step 1: Sample & filter NormBank, build items
│   ├── item_generator.py         # Step 2: GPT-assisted A/B/C item generation
│   ├── inference_engine.py       # Step 3: Run all models across all levels
│   ├── evaluator.py              # Step 4: Compute metrics & McNemar's test
│   ├── visualizer.py             # Step 5: Generate figures & tables
│   └── utils.py                  # Shared utilities
│
├── results/                      # Populated after running evaluator.py
│   ├── metrics_summary.json
│   ├── per_item_results.jsonl
│   └── figures/
│
├── notebooks/
│   └── error_analysis.ipynb      # Qualitative error analysis
│
└── docs/
    ├── dataset_schema.md         # Full JSONL schema documentation
    └── prompt_templates.md       # All prompts used (frozen before inference)
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up API keys
```bash
cp .env.example .env
# Edit .env with your actual keys
```

### 3. Place NormBank data
Download NormBank from https://github.com/SALT-NLP/normbank and place the CSV at:
```
data/raw/normbank.csv
```

### 4. Run the pipeline
```bash
# Step 1: Filter NormBank and sample 40 violations + 20 foils
python src/dataset_constructor.py

# Step 2: Generate A/B/C items (uses GPT-4 to write dialogues)
python src/item_generator.py

# Step 3: Run inference across all models and levels
python src/inference_engine.py

# Step 4: Compute metrics and McNemar's test
python src/evaluator.py

# Step 5: Generate figures
python src/visualizer.py
```

---

## Three-Level Design

| Level | Name | Description | Example |
|-------|------|-------------|---------|
| A | Abstract | Direct yes/no question, no actors | "A colleague presents another person's idea as their own. Is this a social norm violation?" |
| B | Minimal | One sentence with named actors + setting | "In a team meeting, Bob told the manager the new design idea was his, even though Alice had proposed it." |
| C | Situated | 4-turn natural conversation, violation in turn 3 | Full dialogue; model must detect without being cued |

---

## Models Evaluated

- GPT-4o (OpenAI)
- Claude Sonnet 4.6 (Anthropic) — *see disclosure in paper*
- Gemini 1.5 Pro (Google)
- DeepSeek-V2.5 (DeepSeek)

---

## Metrics

- **Primary:** Macro-F1 (handles 2:1 class imbalance)
- **Secondary:** Accuracy, per-class F1
- **Gap Score:** ΔAcc = Acc_A − Acc_C, ΔF1 = MF1_A − MF1_C
- **Significance:** McNemar's χ² test (paired binary, α=0.05)

---

## Norm Categories

- Politeness (10 violations, 5 foils)
- Fairness (10 violations, 5 foils)
- Honesty (10 violations, 5 foils)
- Authority (10 violations, 5 foils)

---

## Disclosure

Claude Sonnet 4.6 belongs to the same model family as the AI assistant used during manuscript development. Mitigations: prompt templates frozen before inference; Level C conversations cross-reviewed by both authors; results disaggregated by model; this disclosure in the paper's Limitations section.

---

## Citation

If you use this framework, please cite NormBank:
```
@inproceedings{ziems2023normbank,
  title={NormBank: A Knowledge Bank of Situational Social Norms},
  author={Ziems, Caleb and others},
  booktitle={ACL},
  year={2023}
}
```
