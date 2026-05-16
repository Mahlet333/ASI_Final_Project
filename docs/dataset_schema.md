# Dataset Schema Documentation

## sampled_norms.jsonl

One record per norm item (40 violations + 20 foils = 60 total).

```json
{
  "item_id":      "POL-V-001",          // {CAT3}-{V|F}-{IDX3}
  "category":     "politeness",         // politeness | fairness | honesty | authority
  "is_violation": true,                 // true = violation, false = foil
  "norm_text":    "interrupting someone while they are speaking",
  "situation":    "team meeting",
  "role":         "employee",
  "raw_row":      { ... }              // original NormBank row
}
```

## items_levelA.jsonl

```json
{
  "item_id":      "POL-V-001",
  "level":        "A",
  "category":     "politeness",
  "is_violation": true,
  "prompt":       "Someone is interrupting someone while they are speaking. Is this a social norm violation?",
  "norm_text":    "interrupting someone while they are speaking",
  "created_at":   "2026-04-01T12:00:00Z"
}
```

## items_levelB.jsonl

```json
{
  "item_id":      "POL-V-001",
  "level":        "B",
  "category":     "politeness",
  "is_violation": true,
  "prompt":       "In a team meeting, Alice was interrupting someone while they are speaking, affecting Bob.",
  "actors":       {"primary": "Alice", "secondary": "Bob"},
  "norm_text":    "interrupting someone while they are speaking",
  "created_at":   "2026-04-01T12:00:00Z"
}
```

## items_levelC.jsonl

```json
{
  "item_id":          "POL-V-001",
  "level":            "C",
  "category":         "politeness",
  "is_violation":     true,
  "dialogue": [
    {"speaker": "Alice",   "text": "I think we should use a blue colour scheme."},
    {"speaker": "Bob",     "text": "Good point. Let me finish reviewing the options."},
    {"speaker": "Alice",   "text": "Actually — so as I was saying, the blue scheme is the right call."},
    {"speaker": "Manager", "text": "Alright, we can finalise the palette in the next meeting."}
  ],
  "speakers":           ["Alice", "Bob", "Manager"],
  "violation_turn":     3,
  "violation_speaker":  "Alice",
  "annotation":         "Alice interrupts Bob mid-sentence in turn 3.",
  "norm_text":          "interrupting someone while they are speaking",
  "created_at":         "2026-04-01T12:00:00Z",
  "fallback":           false
}
```

## api_outputs/{model}_level{L}_run{N}.jsonl

Raw API output — NEVER overwrite these files.

Level A/B:
```json
{
  "item_id":      "POL-V-001",
  "level":        "A",
  "model":        "gpt4o",
  "is_violation": true,
  "raw_response": "{\"violation\": true}",
  "judgment":     true,
  "meta":         {"finish_reason": "stop", "usage": {...}},
  "run":          1,
  "recorded_at":  "2026-04-01T12:05:00Z"
}
```

Level C (two calls):
```json
{
  "item_id":          "POL-V-001",
  "level":            "C",
  "model":            "gpt4o",
  "is_violation":     true,
  "raw_response_c1":  "{\"violation\": true}",
  "raw_response_c2":  "{\"explanation\": \"Alice interrupts Bob while he is still speaking.\"}",
  "judgment":         true,
  "explanation":      "Alice interrupts Bob while he is still speaking.",
  "meta_c1":          {...},
  "meta_c2":          {...},
  "run":              1,
  "recorded_at":      "2026-04-01T12:05:30Z"
}
```

## parsed/{model}_level{L}.jsonl

Aggregated across N_RUNS by majority vote.

```json
{
  "item_id":         "POL-V-001",
  "level":           "C",
  "model":           "gpt4o",
  "is_violation":    true,
  "judgments":       [true, true, false],
  "judgment_final":  true,
  "judgment_runs":   [true, true, false],
  "null_runs":       0,
  "explanations":    ["Alice interrupts Bob..."]
}
```

## IRR annotation file (manual)

Create `data/constructed/irr_annotations.csv`:

```csv
item_id,annotator1_label,annotator2_label
POL-V-001,1,1
POL-V-002,1,0
...
```

- `1` = violation present
- `0` = no violation (foil / norm-compliant)

Both annotators label all 60 Level C items independently before comparing.
Target: Cohen's κ ≥ 0.70.
