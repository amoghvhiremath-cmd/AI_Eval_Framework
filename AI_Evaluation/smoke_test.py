"""
smoke_test.py — Verify that OPENAI_API_KEY, model access, and network all work.

Makes a single trivial call to gpt-4o-mini and prints a clear success or
failure message.  Run this before any full eval run to surface credential
issues cheaply.

Usage
-----
    python smoke_test.py
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("FAIL: OPENAI_API_KEY is not set in the environment.")
        sys.exit(1)

    print("OPENAI_API_KEY found.  Making a test call to gpt-4o-mini …")

    try:
        # Import here so a missing package gives a clear error message
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke([HumanMessage(content='Reply with the single word "ok".')])
        reply = response.content.strip()
        print(f"Model replied: {reply!r}")
        print("SUCCESS: API key, model access, and network are all working.")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
