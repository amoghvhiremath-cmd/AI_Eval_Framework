"""
metrics.py — RAGAS metrics and custom semantic task-set F1.

Text-summary metrics (Faithfulness, FactualCorrectness, RubricsScore) are
async; callers must drive them with asyncio.run or from within an async
context.

The custom task_set_f1 function is synchronous — it embeds tasks directly
using the raw OpenAIEmbeddings object.
"""

from __future__ import annotations

import numpy as np
from langchain_openai import OpenAIEmbeddings
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, FactualCorrectness, RubricsScore

# ---------------------------------------------------------------------------
# Coverage rubric (verbatim from the build brief)
# ---------------------------------------------------------------------------

COVERAGE_RUBRIC: dict[str, str] = {
    "score1_description": "Misses most key facts from the reference, or contradicts it.",
    "score2_description": "Captures a minority of key facts; important omissions.",
    "score3_description": "Captures the main point but misses some supporting facts.",
    "score4_description": "Captures nearly all key facts; minor omissions only.",
    "score5_description": "Fully captures every key fact in the reference; concise and coherent.",
}

# ---------------------------------------------------------------------------
# Per-artifact user_input instructions (short, purposeful)
# ---------------------------------------------------------------------------

ARTIFACT_INSTRUCTIONS: dict[str, str] = {
    "ticket_summary": "Summarize this IT service ticket.",
    "comment_summary": "Summarize the comment thread on this ticket.",
    "diagnostics_summary": "Summarize the diagnostics for this ticket.",
    "resolution_summary": "Summarize how this ticket was resolved.",
}

# ---------------------------------------------------------------------------
# Threshold for task-set cosine matching.
# Tune on real data — 0.60–0.70 is a sane starting band for
# text-embedding-3-small.  Increase to be stricter, decrease to be more lenient.
# ---------------------------------------------------------------------------

TASK_MATCH_THRESHOLD: float = 0.65


def build_metrics(
    llm: LangchainLLMWrapper,
    embeddings: LangchainEmbeddingsWrapper,
) -> tuple[Faithfulness, FactualCorrectness, RubricsScore]:
    """
    Instantiate the three RAGAS text-summary metrics, wired to the given judge.

    Returns
    -------
    faithfulness, factual_correctness, coverage_rubric
    """
    faithfulness = Faithfulness(llm=llm)

    factual_correctness = FactualCorrectness(llm=llm, mode="f1")

    coverage_rubric = RubricsScore(llm=llm, rubrics=COVERAGE_RUBRIC)

    return faithfulness, factual_correctness, coverage_rubric


def build_sample(
    artifact_name: str,
    output_text: str,
    reference_text: str,
    grounding_source_text: str,
) -> SingleTurnSample:
    """Build a RAGAS SingleTurnSample for a single text-summary artifact."""
    return SingleTurnSample(
        user_input=ARTIFACT_INSTRUCTIONS[artifact_name],
        response=output_text,
        reference=reference_text,
        retrieved_contexts=[grounding_source_text],
    )


async def score_text_artifact(
    artifact_name: str,
    output_text: str,
    reference_text: str,
    grounding_source_text: str,
    faithfulness: Faithfulness,
    factual_correctness: FactualCorrectness,
    coverage_rubric: RubricsScore,
) -> dict[str, float | str]:
    """
    Compute all three RAGAS metrics for a single text-summary artifact.

    Each metric is called individually so that one failure does not abort the
    others — the failed metric is recorded as an error string.

    Returns a dict with keys: faithfulness, factual_correctness, coverage_rubric.
    """
    sample = build_sample(artifact_name, output_text, reference_text, grounding_source_text)
    results: dict[str, float | str] = {}

    for name, metric in [
        ("faithfulness", faithfulness),
        ("factual_correctness", factual_correctness),
        ("coverage_rubric", coverage_rubric),
    ]:
        try:
            score = await metric.single_turn_ascore(sample)
            results[name] = round(float(score), 4)
        except Exception as exc:  # noqa: BLE001
            results[name] = f"ERROR: {exc}"

    return results


def task_set_f1(
    candidate_tasks: list[str],
    golden_tasks: list[str],
    raw_embeddings: OpenAIEmbeddings,
    threshold: float = TASK_MATCH_THRESHOLD,
) -> dict[str, float | int]:
    """
    Compute semantic task-set precision / recall / F1.

    Each candidate task is matched to the highest-cosine unused golden task.
    A match is counted only if cosine similarity ≥ *threshold*.

    Parameters
    ----------
    candidate_tasks : list[str]
        AI-generated suggested tasks.
    golden_tasks : list[str]
        Human-verified golden tasks.
    raw_embeddings : OpenAIEmbeddings
        Unwrapped OpenAI embeddings for direct embed_documents calls.
    threshold : float
        Cosine similarity threshold for a valid match.
        NOTE: tune this on real data — 0.60–0.70 is recommended for
        text-embedding-3-small; increase for stricter matching.

    Returns
    -------
    dict with keys: precision, recall, f1, matched  (all rounded to 4 dp).
    """
    zero = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0}
    if not candidate_tasks or not golden_tasks:
        return zero

    # Embed both task lists
    all_texts = candidate_tasks + golden_tasks
    all_vecs = raw_embeddings.embed_documents(all_texts)
    n_cand = len(candidate_tasks)

    cand_vecs = np.array(all_vecs[:n_cand], dtype=float)
    gold_vecs = np.array(all_vecs[n_cand:], dtype=float)

    # Normalise for cosine similarity via dot product
    cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
    gold_norms = np.linalg.norm(gold_vecs, axis=1, keepdims=True)
    cand_vecs = np.divide(cand_vecs, cand_norms, where=cand_norms != 0)
    gold_vecs = np.divide(gold_vecs, gold_norms, where=gold_norms != 0)

    # Greedy one-to-one matching
    similarity_matrix = cand_vecs @ gold_vecs.T  # shape: (n_cand, n_gold)
    used_gold: set[int] = set()
    matched = 0

    for cand_idx in range(len(candidate_tasks)):
        sims = similarity_matrix[cand_idx].copy()
        # Mask already-matched golden tasks
        for used in used_gold:
            sims[used] = -1.0
        best_gold_idx = int(np.argmax(sims))
        if sims[best_gold_idx] >= threshold:
            matched += 1
            used_gold.add(best_gold_idx)

    precision = matched / len(candidate_tasks)
    recall = matched / len(golden_tasks)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": matched,
    }
