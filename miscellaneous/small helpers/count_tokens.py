"""
count_tokens.py

Counts the number of tokens for a given input using the DeepSeek R1 Distill 14B tokenizer.

How input is provided (no terminal args needed):
- Preferred: set INPUT_TEXT below.
- Fallback: create a text file named "input.txt" next to this script.

Usage:
    python count_tokens.py

Output:
    Prints a single integer (token count) to stdout.

Notes:
- Defaults to model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B". Override with env MODEL_NAME.
- If /dev/shm exists, the script will prefer it for HF caches to avoid filling disk volumes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# Optional inline input. If empty, the script will try ./input.txt
INPUT_TEXT: str = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3)))))))))\). Wow, that's a lot of nested parentheses! It looks like a recursive expression, where each layer is 3 times (1 plus something). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let's identify the innermost part.
Looking at the expression, the innermost parentheses is the last one: (1 + 3). Let me compute that first.
1 + 3 = 4.
Okay, so replacing that innermost part, the expression becomes: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(4)))))))))\).
Wait, that's a bit confusing. Maybe I should label each layer as I go. Let me number them from the innermost as Layer 1 to the outermost as Layer 9.
Layer 1: (1 + 3) = 4.
Then Layer 2: 3(1 + Layer 1) = 3*(1 + 4) = 3*5 = 15."""


def _prepare_hf_cache_env() -> None:
    """Prefer RAM-backed cache on systems with /dev/shm to avoid disk usage."""
    if os.path.isdir("/dev/shm"):
        ram_cache = "/dev/shm/hf"
        # Only set if not already set by the environment
        os.environ.setdefault("HF_HOME", ram_cache)
        os.environ.setdefault("HF_HUB_CACHE", ram_cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", ram_cache)


def _load_input() -> str:
    """Return input text from INPUT_TEXT or ./input.txt.

    Priority:
    1) Non-empty INPUT_TEXT constant in this file
    2) A sibling file named input.txt
    """
    if INPUT_TEXT.strip():
        return INPUT_TEXT

    candidate = Path(__file__).with_name("input.txt")
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")

    print(
        "No input provided. Set INPUT_TEXT in the script or create an 'input.txt' file.",
        file=sys.stderr,
    )
    sys.exit(2)


def count_tokens(text: str) -> int:
    """Count tokens using the DeepSeek R1 Distill 14B tokenizer.

    The default model is assumed to be 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'.
    Set MODEL_NAME env var to change this.
    """
    _prepare_hf_cache_env()

    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time
        print(
            "Transformers is required. Install with: pip install transformers --upgrade",
            file=sys.stderr,
        )
        raise

    model_name = os.environ.get(
        "MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    )

    cache_dir = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HF_HOME")
        or os.environ.get("TRANSFORMERS_CACHE")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        cache_dir=cache_dir,
    )

    # Do not add special tokens for plain text counting
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


def main() -> None:
    text = _load_input()
    total = count_tokens(text)
    # Print only the integer count
    print(total)


if __name__ == "__main__":
    main()


