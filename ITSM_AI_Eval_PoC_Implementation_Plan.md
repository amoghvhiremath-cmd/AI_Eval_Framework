# ITSM AI Eval — PoC Implementation Plan

**Scope:** Local proof-of-concept. Prove that a RAGAS + OpenAI pipeline produces
believable quality scores for the AI-generated content in the ITSM app, on a
small verified dataset, before investing in the full CI/CD framework.

**Status of inputs:** Golden dataset (10–20 verified tickets) is in progress.
This plan is sequenced so framework build happens *in parallel* with dataset
prep — we build against a 1–2 record mock file, then swap the real data in.

---

## 1. Objective & success criteria

The PoC succeeds when all three are true:

1. The pipeline runs end to end and prints per-ticket + aggregate scores for all
   5 artifacts.
2. On a manual spot-check of 2–3 tickets, the scores match human gut judgment of
   the output quality (a bad summary scores low, a good one scores high).
3. Total OpenAI spend for a full run stays well under $1.

This is a *plumbing and believability* test, not a release gate. No pass/fail
thresholds, no CI, no production monitoring — those come later.

---

## 2. In scope / out of scope

**In scope**
- Local run on one machine, invoked manually from the terminal.
- Single judge: OpenAI `gpt-4o-mini` (temperature 0).
- Embeddings: OpenAI `text-embedding-3-small`.
- One combined JSON file holding all data.
- All 5 artifacts (see §4).
- Reference-based + reference-free metrics (see §5).
- Console report: per-ticket scores + aggregate means.

**Out of scope (deliberately, for now)**
- CI/CD gate, threshold config, PR blocking.
- Two-judge setup (GPT-5 + Claude) and judge-agreement tracking.
- AWS Bedrock wiring (kept for the production build; swap-in is a one-line model
  change later).
- Reference-free production/flash tier and the feedback loop.

---

## 3. Prerequisites

- Python 3.10+ and a virtual environment.
- OpenAI API key with credits, set as `OPENAI_API_KEY`.
- The data contract below understood by whoever prepares the dataset.

---

## 4. The 5 artifacts

| # | Artifact | What the AI generates | Grounding source (for faithfulness) |
|---|----------|-----------------------|--------------------------------------|
| 1 | `ticket_summary` | Summary of the ticket | ticket description |
| 2 | `comment_summary` | Summary of the comment thread | comments |
| 3 | `diagnostics_summary` | Summary of diagnostics | diagnostics |
| 4 | `resolution_summary` | How the ticket was resolved | comments |
| 5 | `suggested_tasks` | Step-by-step resolution tasks | description (scored vs golden tasks) |

---

## 5. Data contract (the JSON file)

One JSON array. Each record = one ticket with **three parts**:

- `source` — the inputs the AI saw.
- `references` — your verified golden answer (the answer key).
- `output` — the AI-generated answer to be scored (the candidate).

```json
[
  {
    "ticket_id": "INC-10492",
    "category": "identity_and_access",
    "priority": "P2",
    "source": {
      "description": "…raw ticket text…",
      "comments": ["…", "…"],
      "diagnostics": "…raw diagnostic data…"
    },
    "references": {
      "ticket_summary": "…verified golden summary…",
      "comment_summary": "…",
      "diagnostics_summary": "…",
      "resolution_summary": "…",
      "suggested_tasks": ["…", "…"]
    },
    "output": {
      "ticket_summary": "…AI-generated summary to test…",
      "comment_summary": "…",
      "diagnostics_summary": "…",
      "resolution_summary": "…",
      "suggested_tasks": ["…", "…"]
    }
  }
]
```

> **Critical rule:** `output` must be a real, separate AI generation — NOT a copy
> of `references`. If the two are identical, every score is a perfect 100% and
> the PoC measures nothing. `source` + `references` come from your golden
> dataset; `output` is the AI result you want to evaluate.

---

## 6. Metric plan

| Artifact | Metrics | Type | Needs reference? |
|----------|---------|------|------------------|
| ticket_summary | Faithfulness, Factual correctness, Coverage rubric (1–5) | RAGAS | Correctness + coverage: yes |
| comment_summary | Faithfulness, Factual correctness, Coverage rubric | RAGAS | yes |
| diagnostics_summary | Faithfulness, Factual correctness, Coverage rubric | RAGAS | yes |
| resolution_summary | Faithfulness, Factual correctness, Coverage rubric | RAGAS | yes |
| suggested_tasks | Semantic task-set F1 (precision / recall / F1) | Custom (embeddings) | yes |

Metric meanings, plainly:
- **Faithfulness** — did the output invent anything the source doesn't support?
  (grounding / anti-hallucination; uses source, not reference)
- **Factual correctness** — does the output match the golden answer? (F1 over
  decomposed claims)
- **Coverage rubric** — how fully it captures the golden answer, scored 1–5.
- **Task-set F1** — each suggested task is embedded and matched by meaning to the
  golden task list; rewards correct tasks worded differently, penalises missing
  or invented tasks.

---

## 7. Build steps (sequenced)

### Step 1 — Environment & smoke test (~15 min)
Create venv, install pinned deps, set `OPENAI_API_KEY`. Run a 3-line smoke test
that makes one trivial judge call to confirm the key and model work. Do this
*before* any full run so credential issues surface for free.

### Step 2 — Data contract + validator (~30 min)
Implement the schema (§5) as a loader that validates the JSON and fails loudly
on malformed records. Ship a `data/sample.json` with 1–2 mock tickets so the
framework can be built and tested before the real dataset lands.

### Step 3 — Judge & embeddings wiring (~20 min)
Wire OpenAI `gpt-4o-mini` (temperature 0) and `text-embedding-3-small`, wrapped
for RAGAS. Isolated so swapping to a frontier judge or Bedrock later is one line.

### Step 4 — Vertical slice: one metric, one ticket (~30 min)
Get a single Faithfulness score for a single ticket and print it. This de-risks
the whole RAGAS-to-OpenAI path on the smallest surface. If this works, the rest
is repetition.

### Step 5 — All metrics, all 5 artifacts (~1–2 hrs)
Add factual correctness, the coverage rubric, and the custom task-set F1. Loop
over all 5 artifacts per ticket. Handle missing fields gracefully (skip, don't
crash).

### Step 6 — Runner + console report (~45 min)
Aggregate scores into per-ticket and dataset-level means, grouped by artifact.
Print a readable table. Optionally dump `report.json`.

### Step 7 — Swap in real data + eyeball (when dataset ready)
Point the runner at `data/golden.json`. Run the full set. Spot-check 2–3 tickets
by hand against the scores. If the numbers are believable, the PoC is done.

---

## 8. Project structure

```
itsm-eval-poc/
  README.md
  requirements.txt
  smoke_test.py
  data/
    sample.json          # 1–2 mock records to build against
    golden.json          # your real 10–20, added when ready
  src/
    __init__.py
    schema.py            # data contract + validator
    judge.py             # OpenAI judge + embeddings for RAGAS
    metrics.py           # RAGAS metrics + custom task-set F1
    runner.py            # scoring + aggregation
    cli.py               # entrypoint
```

---

## 9. How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python smoke_test.py                          # confirm key + model
python -m src.cli --data data/sample.json     # build/test on mock
python -m src.cli --data data/golden.json     # real run when ready
```

---

## 10. Cost estimate

Judge is `gpt-4o-mini`. Faithfulness and factual correctness each make several
small calls per artifact (they decompose text into claims), so a full run of
~20 tickets × 5 artifacts is on the order of a few cents to roughly $0.50 —
comfortably inside $5. Embeddings for the task metric are effectively free at
this scale. A frontier judge would cost noticeably more; that swap is for later.

---

## 11. Risks & gotchas

- **Candidate == reference.** The #1 trap. If `output` is a copy of
  `references`, scores are meaningless perfect 100s. Enforce separate text
  during data prep.
- **RAGAS API drift.** Its metric API shifts between minor versions — pin the
  version in `requirements.txt`; upgrades may need small import fixes in
  `metrics.py`.
- **Model IDs.** Confirm the exact OpenAI model names against your account.
- **Weak-judge caveat.** `gpt-4o-mini` is a lightweight judge — fine for proving
  plumbing, not authoritative. Don't over-read absolute values; watch relative
  ranking (good vs bad outputs) instead.
- **Small N.** 10–20 tickets is enough to validate the approach, not to make
  statistical claims about model quality.

---

## 12. After the PoC (not now)

Once the numbers are believable: add threshold gating + a CI entrypoint,
introduce the second (frontier) judge and track judge-vs-judge agreement, wire
AWS Bedrock as an alternative judge/embeddings backend, and build the
reference-free production tier. All are additive to this PoC's structure.
