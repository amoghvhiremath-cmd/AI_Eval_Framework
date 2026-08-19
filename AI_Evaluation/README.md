# AI Eval PoC

A local proof-of-concept that evaluates the quality of AI-generated content produced by an ITSM (IT Service Management) application, using [RAGAS](https://docs.ragas.io/) with an OpenAI judge.

---

## What it does

The app uses LLMs (in production) to auto-generate five artifacts per ticket:

| Artifact | Description |
|---|---|
| `ticket_summary` | Summary of the ticket |
| `comment_summary` | Summary of the comment thread |
| `diagnostics_summary` | Summary of the diagnostics |
| `resolution_summary` | How the ticket was resolved |
| `suggested_tasks` | Step-by-step resolution tasks |

This tool scores each artifact against a **human-verified golden answer** and prints per-ticket and aggregate scores. It is a plumbing and believability test — not a release gate.

### Metrics

| Artifact | Metric | What it measures |
|---|---|---|
| All summaries | **Faithfulness** (0–1) | Did the output hallucinate anything not in the source? |
| All summaries | **Factual correctness** (0–1) | Does the output match the golden answer (claim F1)? |
| All summaries | **Coverage rubric** (1–5) | How fully does the output capture the golden answer? |
| `suggested_tasks` | **Semantic task-set F1** | Each task matched by embedding cosine similarity to the golden task list |

---

## Prerequisites

- Python 3.10 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to `gpt-4o-mini` and `text-embedding-3-small`
- Credits on your OpenAI account (a full run costs a few cents to ~$0.50)

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
# On Windows (PowerShell):
$env:OPENAI_API_KEY = "sk-..."
# On macOS/Linux:
export OPENAI_API_KEY="sk-..."
```

---

## Running

### Step 1 — Smoke test (confirm key + model work)

```bash
python smoke_test.py
```

Expected output:
```
OPENAI_API_KEY found.  Making a test call to gpt-4o-mini …
Model replied: 'ok'
SUCCESS: API key, model access, and network are all working.
```

### Step 2 — Run on the mock dataset

```bash
python -m src.cli --data data/sample.json
```

This runs against the two included mock tickets and prints per-ticket + aggregate scores.

### Step 3 — Run on the real golden dataset (when ready)

```bash
python -m src.cli --data data/golden.json
```

`data/golden.json` follows the exact same schema as `data/sample.json` — just add your verified 10–20 ticket records there.

### Optional: save the full JSON report

```bash
python -m src.cli --data data/sample.json --report report.json
```

---

## Data format

`data/golden.json` should be a JSON array where each record has this shape:

```json
{
  "ticket_id": "INC-XXXXX",
  "category": "...",
  "priority": "P1",
  "source": {
    "description": "raw ticket text",
    "comments": ["comment 1", "comment 2"],
    "diagnostics": "raw diagnostic data"
  },
  "references": {
    "ticket_summary": "verified golden summary",
    "comment_summary": "...",
    "diagnostics_summary": "...",
    "resolution_summary": "...",
    "suggested_tasks": ["task 1", "task 2"]
  },
  "output": {
    "ticket_summary": "AI-generated summary to score",
    "comment_summary": "...",
    "diagnostics_summary": "...",
    "resolution_summary": "...",
    "suggested_tasks": ["task 1", "task 2"]
  }
}
```

> **Important:** `output` must be a real, separate AI generation — not a copy of `references`. If the two are identical, every score will be perfect and the eval measures nothing.

All artifact fields are optional (`null`). Missing fields are skipped gracefully.

---

## Configuration

Model IDs are constants in `src/judge.py` and can be overridden via environment variables:

| Env var | Default | Controls |
|---|---|---|
| `EVAL_JUDGE_MODEL` | `gpt-4o-mini` | LLM judge for RAGAS metrics |
| `EVAL_EMBED_MODEL` | `text-embedding-3-small` | Embeddings for task-set F1 |

The task-match cosine threshold (default `0.65`) is a constant in `src/metrics.py` — tune it on real data (0.60–0.70 is a sane band for `text-embedding-3-small`).

---

## Project structure

```
AI_Evaluation/
  README.md

  requirements.txt
  smoke_test.py          # API key + model connectivity check
  data/
    sample.json          # 2 mock records for development
    golden.json          # your real 10–20 records (not included)
  src/
    __init__.py
    schema.py            # Pydantic data contract + loader
    judge.py             # OpenAI judge + embeddings wrappers
    metrics.py           # RAGAS metrics + custom task-set F1
    runner.py            # scoring loop + aggregation
    cli.py               # command-line entrypoint
```
