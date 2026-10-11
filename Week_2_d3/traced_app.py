"""
Week 2 · Day 3 — Trace a LangChain app with LangSmith.

Before running:
  1. Install dependencies: python -m pip install -r requirements.txt
  2. Set OPENAI_API_KEY, LANGSMITH_TRACING, LANGSMITH_API_KEY,
     and LANGSMITH_PROJECT in your terminal environment.
  3. Run: python traced_app.py

This script does not configure tracing in code. LangSmith tracing is enabled
through environment variables only.
"""

import os
import sys
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

PROJECT_NAME = "week2_d4_ls_tracing"
MODEL_NAME = "gpt-4o-mini"

PROMPTS = [
    (
        "Prompt 1 — short",
        "In one sentence, explain what a Python list is."
    ),
    (
        "Prompt 2 — medium",
        "Explain the difference between a Python list and a tuple. "
        "Give one short code example of each and finish with a two-item "
        "comparison."
    ),
    (
        "Prompt 3 — detailed",
        "You are helping a beginner understand a small RAG application. "
        "Explain the following stages in order: document chunking, embedding, "
        "storing vectors in FAISS, retrieving the top 3 relevant chunks, "
        "placing the retrieved text into a prompt, and asking an LLM to answer. "
        "For every stage, state what goes in, what comes out, and one common "
        "mistake. End with a compact numbered flow. Keep the explanation "
        "clear and practical."
    ),
]


def check_environment() -> bool:
    """Check required environment variables without printing secret values."""
    

    required = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY"),
        "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING"),
        "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT"),
        "LANGSMITH_ENDPOINT": os.getenv("LANGSMITH_ENDPOINT")
    }

    missing = [name for name, value in required.items() if not value]
    if missing:
        print("ERROR: These environment variables are missing or empty:")
        for name in missing:
            print(f"  - {name}")
        print("\nSet them in your terminal, then run the script again.")
        return False

    tracing_value = required["LANGSMITH_TRACING"].strip().lower()
    if tracing_value not in {"true", "1", "yes", "on"}:
        print(
            "ERROR: LANGSMITH_TRACING is set, but not to a true value. "
            'Set it to "true".'
        )
        return False

    project = required["LANGSMITH_PROJECT"]
    print("Preflight check passed.")
    print("  OpenAI API key: present (value hidden)")
    print("  LangSmith API key: present (value hidden)")
    print(f"  LangSmith tracing: {tracing_value}")
    print(f"  LangSmith project: {project}")
    print("  Model:", MODEL_NAME)
    print("\nSending 3 prompts...\n")
    return True


def main() -> int:
    if not check_environment():
        return 1

    # LangSmith tracing is picked up automatically from environment variables.
    # No LangSmith tracing code is needed here.
    llm = ChatOpenAI(model=MODEL_NAME)

    
    for label, prompt in PROMPTS:
        start_time = time.perf_counter()
        print("=" * 72)
        print(label)
        print("Prompt:", prompt)
        try:
            response = llm.invoke(prompt)
            print("\nAnswer:")
            end_time = time.perf_counter() 
            print(response.content)
            latency = end_time - start_time 
            usage = response.usage_metadata
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)  

            print(f"Latency       : {latency:.2f} seconds")
            print(f"Input tokens  : {input_tokens}")
            print(f"Output tokens : {output_tokens}")
            print(f"Total tokens  : {total_tokens}")          
        except Exception as exc:
            print(f"\nERROR while calling the model: {exc}", file=sys.stderr)
            print(
                "\nCheck your OpenAI API key, account/API billing, network, "
                "and installed package versions.",
                file=sys.stderr,
            )
            return 1
      
    
    print("\n" + "=" * 72)
    print("Finished. Check the LangSmith project shown above for the 3 runs.")
    print(
        "Open a run to inspect the exact input, output, latency, and token "
        "usage (when reported by the model/provider)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

        

