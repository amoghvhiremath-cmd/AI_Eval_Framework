"""
judge.py — OpenAI judge LLM and embeddings, wrapped for RAGAS.

Model IDs are module-level constants (overridable via env vars) so swapping
to a frontier judge or an alternative provider is a one-line change.

The module never hard-codes secrets; ChatOpenAI / OpenAIEmbeddings read
OPENAI_API_KEY from the environment automatically.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

# ---------------------------------------------------------------------------
# Constants — swap here to change models for the whole pipeline
# ---------------------------------------------------------------------------

JUDGE_MODEL: str = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
EMBED_MODEL: str = os.getenv("EVAL_EMBED_MODEL", "text-embedding-3-small")


def build_judge() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper, OpenAIEmbeddings]:
    """
    Build and return the judge LLM, the RAGAS-wrapped embeddings, and the
    raw OpenAIEmbeddings (used directly for the custom task-set F1).

    Returns
    -------
    llm : LangchainLLMWrapper
        RAGAS-compatible wrapper around ChatOpenAI (temperature 0).
    embeddings : LangchainEmbeddingsWrapper
        RAGAS-compatible wrapper around OpenAIEmbeddings.
    raw_embeddings : OpenAIEmbeddings
        Unwrapped embeddings for direct `embed_documents` calls (task F1).
    """
    raw_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    llm = LangchainLLMWrapper(raw_llm)

    raw_embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

    return llm, embeddings, raw_embeddings
