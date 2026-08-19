# ITSM AI Eval PoC — Build Brief for Cursor

You are building a **local proof-of-concept** that evaluates the quality of
AI-generated content produced by an ITSM (IT Service Management) application.
Read this whole document before writing code. Build strictly to this spec — do
not add scope (no CI, no AWS, no web UI, no database).

---

## 1. Background & context

An ITSM application handles IT incidents, service requests, and queries. It uses
LLMs (Claude, in production) to auto-generate, per ticket:

- **ticket_summary** — a summary of the ticket
- **comment_summary** — a summary of the ticket's comment thread
- **diagnostics_summary** — a summary of the diagnostics
- **resolution_summary** — how the ticket was resolved
- **suggested_tasks** — a step-by-step list of tasks to resolve the ticket

We need to measure how good those generated outputs are. This PoC scores each
artifact against a **human-verified golden answer** using RAGAS with an OpenAI
judge, on a small dataset (10–20 tickets), and prints the scores. The goal is to
prove the approach works and produces believable numbers — **not** to build a
release gate. No pass/fail thresholds, no CI.

### Core concept the code must respect
Every scored record has three parts per artifact:
- `source` — the inputs the AI saw (what the ticket contained).
- `references` — the human-verified correct answer (the answer key).
- `output` — the AI-generated answer being tested (the candidate).

`output` and `references` are **different text**. The code never assumes they are
equal; it scores `output` against `references` (and, for grounding, against
`source`).

---

## 2. Tech stack

| Concern | Choice | Notes |
|---------|--------|-------|
| Language | Python 3.10+ | type hints throughout |
| Eval framework | **RAGAS** (pinned `0.2.14`) | metric API shifts between versions — keep pinned |
| Judge LLM | OpenAI **`gpt-4o-mini`**, temperature 0 | lightweight judge; swappable via one constant |
| Embeddings | OpenAI **`text-embedding-3-small`** | for the custom task metric |
| LLM/embeddings glue | `langchain-openai` + RAGAS wrappers | see §6 |
| Data validation | `pydantic` v2 | fail loud on malformed input |
| Numerics | `numpy` | cosine similarity for task metric |
| Config / secrets | `OPENAI_API_KEY` env var only | never hardcode keys |

Do **not** add: AWS/Bedrock, CI config, Flask/FastAPI, a database, or browser
storage. Those are future phases and out of scope.

### `requirements.txt`
```
ragas==0.2.14
langchain-openai==0.2.14
pydantic>=2.6
numpy>=1.26
```

---

## 3. Project structure to create

```
itsm-eval-poc/
  README.md                # setup + run instructions (you write this)
  requirements.txt
  smoke_test.py            # confirms OPENAI_API_KEY + model work
  data/
    sample.json            # 2 mock records (provided in §7) to build against
    golden.json            # real 10–20 records added later by us — do not create
  src/
    __init__.py
    schema.py              # pydantic data contract + JSON loader/validator
    judge.py               # OpenAI judge + embeddings, wrapped for RAGAS
    metrics.py             # RAGAS metrics + custom semantic task-set F1
    runner.py              # scoring loop + aggregation
    cli.py                 # command-line entrypoint
```

---

## 4. Data contract (§5 of schema.py)

Input file = a JSON **array** of ticket records. Each record:

```json
{
  "ticket_id": "INC-10492",
  "category": "identity_and_access",
  "priority": "P2",
  "source": {
    "description": "string",
    "comments": ["string", "..."],
    "diagnostics": "string"
  },
  "references": {
    "ticket_summary": "string | null",
    "comment_summary": "string | null",
    "diagnostics_summary": "string | null",
    "resolution_summary": "string | null",
    "suggested_tasks": ["string", "..."]
  },
  "output": {
    "ticket_summary": "string | null",
    "comment_summary": "string | null",
    "diagnostics_summary": "string | null",
    "resolution_summary": "string | null",
    "suggested_tasks": ["string", "..."]
  }
}
```

Rules for `schema.py`:
- Define pydantic models `Source`, `Artifacts` (used for both references and
  output), and `TicketRecord`. All artifact fields are optional (`None`).
- `load_dataset(path) -> list[TicketRecord]` reads the JSON array and validates
  each record. On any validation error, raise with the record index / ticket_id
  so it's obvious which record is bad. Fail loud — never silently skip.

### Artifact → grounding-source map (used for faithfulness)
| Artifact | Grounding source field |
|----------|------------------------|
| ticket_summary | `source.description` |
| comment_summary | `source.comments` (joined with newlines) |
| diagnostics_summary | `source.diagnostics` |
| resolution_summary | `source.comments` (joined) |
| suggested_tasks | not used (scored vs golden tasks only) |

---

## 5. Metrics (metrics.py)

### Text artifacts (the four summaries)
For each summary artifact where BOTH `output` and `references` are present,
compute three metrics:

1. **Faithfulness** — RAGAS `Faithfulness`. Measures grounding of `output`
   against the artifact's **source** (not the reference). Uses
   `retrieved_contexts = [grounding_source_text]`. Range 0–1.
2. **Factual correctness** — RAGAS `FactualCorrectness(mode="f1")`. Measures
   `output` vs `references`. Range 0–1.
3. **Coverage rubric** — RAGAS `RubricsScore` with the rubric below. Measures how
   fully `output` captures `references`. Range 1–5.

Coverage rubric to use verbatim:
```python
COVERAGE_RUBRIC = {
    "score1_description": "Misses most key facts from the reference, or contradicts it.",
    "score2_description": "Captures a minority of key facts; important omissions.",
    "score3_description": "Captures the main point but misses some supporting facts.",
    "score4_description": "Captures nearly all key facts; minor omissions only.",
    "score5_description": "Fully captures every key fact in the reference; concise and coherent.",
}
```

RAGAS sample construction (RAGAS 0.2.x):
```python
from ragas.dataset_schema import SingleTurnSample
sample = SingleTurnSample(
    user_input="Summarize this ticket.",      # short instruction per artifact
    response=output_text,
    reference=reference_text,
    retrieved_contexts=[grounding_source_text],
)
score = await metric.single_turn_ascore(sample)   # metrics are async
```
Use one instruction string per artifact (e.g. "Summarize the comment thread on
this ticket." etc.). Metrics are async — the runner drives them with
`asyncio.run`.

### suggested_tasks — custom semantic task-set F1
Implement `task_set_f1(candidate_tasks, golden_tasks, raw_embeddings, threshold=0.65) -> dict`:
- If either list is empty → return precision/recall/f1 = 0.
- Embed both lists via `raw_embeddings.embed_documents(...)`.
- Greedy one-to-one matching: for each candidate, find the highest-cosine unused
  golden task; count a match if cosine ≥ `threshold`.
- `precision = matched / len(candidate)`, `recall = matched / len(golden)`,
  `f1 = harmonic mean`.
- Return `{"precision", "recall", "f1", "matched"}` rounded to 4 dp.
- `threshold` is configurable; note in a comment that it should be tuned on real
  data (0.6–0.7 is a sane starting band for `text-embedding-3-small`).

---

## 6. Judge & embeddings (judge.py)

Wrap OpenAI for RAGAS. Keep model IDs as module-level constants so they can be
swapped in one place.

```python
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EVAL_EMBED_MODEL", "text-embedding-3-small")

def build_judge():
    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    raw_embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    embeddings = LangchainEmbeddingsWrapper(raw_embeddings)
    return llm, embeddings, raw_embeddings
```
`ChatOpenAI` / `OpenAIEmbeddings` read `OPENAI_API_KEY` from the environment
automatically. Never pass the key in code.

---

## 7. `data/sample.json` — create with these 2 mock records

Note the `output` text is deliberately *different* from `references` (paraphrased
+ one missing task) so scores are realistic, not perfect.

```json
[
  {
    "ticket_id": "INC-10492",
    "category": "identity_and_access",
    "priority": "P2",
    "source": {
      "description": "Software license activations are failing because an expired license synchronization token blocked newly deployed software from activating, delaying employee onboarding. Affected service: Access Governance Platform; affected CI: Privileged Access Management System.",
      "comments": [
        "Refresh of the license server completed but the problem persists.",
        "Troubleshooting focused on validating assigned credentials for invalid credentials.",
        "Investigation in progress; credential verification is the next diagnostic step."
      ],
      "diagnostics": "Expired or invalid license synchronization token preventing activation. High-confidence match to prior incidents on the same CI. Server refresh did not resolve it; credential validation in progress."
    },
    "references": {
      "ticket_summary": "License activations are failing due to an expired license synchronization token, blocking newly deployed software and delaying onboarding. Affects the Access Governance Platform and Privileged Access Management System.",
      "comment_summary": "A license server refresh did not fix the issue. Troubleshooting then focused on validating credentials. Investigation is ongoing with credential verification as the next step.",
      "diagnostics_summary": "An expired/invalid license synchronization token is blocking activation on the Privileged Access Management System; a server refresh failed and credential validation is underway.",
      "resolution_summary": "Resolved by renewing the expired license synchronization credentials and confirming successful communication with the license server.",
      "suggested_tasks": [
        "Verify the license synchronization token status and expiration",
        "Regenerate the token if expired and validate it is recognized",
        "Validate license server connectivity and review logs for auth/sync errors",
        "Reprocess the failed license activations",
        "Verify resolution and monitor for recurrence"
      ]
    },
    "output": {
      "ticket_summary": "License activation is failing because a license sync token expired, which stopped new software from activating and slowed onboarding on the Access Governance Platform.",
      "comment_summary": "Refreshing the license server did not help. The team then checked credentials. Work is ongoing, with credential verification next.",
      "diagnostics_summary": "An expired license synchronization token is blocking activation on the Privileged Access Management System; a refresh did not fix it and credentials are being validated.",
      "resolution_summary": "Fixed by renewing the expired license synchronization credentials and confirming the license server responded.",
      "suggested_tasks": [
        "Check the license sync token status and expiration",
        "Regenerate the token if it has expired and confirm it is recognized",
        "Check license server connectivity and review authentication/sync logs",
        "Re-run the failed license activations"
      ]
    }
  },
  {
    "ticket_id": "INC-10510",
    "category": "network",
    "priority": "P3",
    "source": {
      "description": "Users on the 3rd floor report intermittent WiFi drops since this morning. Affected CI: Floor-3 access point AP-3B.",
      "comments": [
        "Confirmed AP-3B showing repeated deauth events in the controller logs.",
        "Rebooted AP-3B; drops reduced but not eliminated.",
        "Replaced AP-3B with a spare unit; no drops observed for 2 hours."
      ],
      "diagnostics": "Controller logs show repeated deauthentication events isolated to AP-3B. Signal strength normal on adjacent APs. Points to failing access point hardware."
    },
    "references": {
      "ticket_summary": "Intermittent WiFi drops on the 3rd floor since morning, traced to access point AP-3B.",
      "comment_summary": "AP-3B showed repeated deauth events. A reboot reduced but did not stop drops; replacing the unit with a spare eliminated them.",
      "diagnostics_summary": "Repeated deauthentication events isolated to AP-3B with normal signal on adjacent APs indicate failing access point hardware.",
      "resolution_summary": "Resolved by replacing the failing access point AP-3B with a spare unit, after which no drops were observed.",
      "suggested_tasks": [
        "Review controller logs for the affected access point",
        "Reboot the access point and observe",
        "If drops persist, replace the access point with a spare",
        "Monitor for recurrence after replacement"
      ]
    },
    "output": {
      "ticket_summary": "Third-floor users have had intermittent WiFi drops since the morning; the issue is linked to access point AP-3B.",
      "comment_summary": "Logs showed deauth events on AP-3B. Rebooting helped only partially, so the AP was swapped for a spare and drops stopped.",
      "diagnostics_summary": "Deauthentication events limited to AP-3B, with adjacent APs normal, suggest the access point hardware is failing.",
      "resolution_summary": "Fixed by swapping the faulty AP-3B for a spare, after which the connection was stable.",
      "suggested_tasks": [
        "Check controller logs for the access point in question",
        "Reboot the access point and watch for drops",
        "Replace the access point with a spare if drops continue",
        "Keep monitoring after the swap"
      ]
    }
  }
]
```

---

## 8. Runner (runner.py)

- `run(path) -> dict`:
  - Load + validate the dataset.
  - Build judge/embeddings once.
  - Instantiate the three text metrics once.
  - For each ticket:
    - For each of the 4 summary artifacts where `output` and `references` both
      exist: build the sample, run the 3 metrics via `single_turn_ascore`
      (async), collect scores. Wrap each metric call in try/except so one
      failure records an error for that metric instead of crashing the run.
    - For `suggested_tasks` where both lists exist: run `task_set_f1`.
  - Return `{"per_ticket": {...}, "aggregate": {...}}`.
- `aggregate`: mean of each numeric metric per artifact across all tickets.
- Use `asyncio.run` to drive the async metric calls.

---

## 9. CLI (cli.py)

```
python -m src.cli --data data/sample.json [--report report.json]
```
- Parse `--data` (required) and optional `--report`.
- Call `runner.run`, print a readable per-artifact aggregate table to stdout, and
  optionally dump the full report as JSON.
- Exit 0 always (this PoC does not gate).

---

## 10. Smoke test (smoke_test.py)

A standalone script that makes ONE trivial `gpt-4o-mini` call (e.g. asks it to
reply "ok") and prints the result, so we can confirm the API key, model access,
and network work before spending on a full run. Print a clear success/failure
message.

---

## 11. README.md (you write it)

Cover: what the tool does, prerequisites (Python 3.10+, `OPENAI_API_KEY`), setup
(venv + `pip install -r requirements.txt`), and the run commands (smoke test,
sample run, and note that `data/golden.json` is supplied later in the same shape
as `data/sample.json`).

---

## 12. Coding conventions

- Type hints on all public functions; short docstrings explaining intent.
- Fail loud on bad input (malformed JSON, missing required source fields).
- Missing *optional* artifact fields are skipped gracefully, not errors.
- No secrets in code — `OPENAI_API_KEY` from env only.
- Keep judge model, embedding model, and task-match threshold as named constants
  so they're trivial to change.
- Round printed scores to 4 decimal places.

---

## 13. Definition of done

1. `pip install -r requirements.txt` succeeds.
2. `python smoke_test.py` confirms the OpenAI key/model work.
3. `python -m src.cli --data data/sample.json` runs end to end and prints
   per-artifact aggregate scores for all 5 artifacts, on both mock tickets,
   without crashing.
4. Scores are non-trivial (not all 0, not all perfect) — because sample `output`
   differs from `references`.
5. Swapping `--data data/golden.json` (same schema, added later) works with no
   code changes.

---

## 14. Explicitly out of scope (do not build)

CI/CD config, threshold gating / pass-fail, AWS Bedrock, a second judge model,
judge-agreement tracking, production/live-trace evaluation, web UI, database,
persistence. These are later phases; keep this PoC minimal.
