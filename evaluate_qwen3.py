from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteriaList
import csv
import datetime
import os
import math


# ===== User inputs =====
PROMPT = "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$"
INTENDED_ANSWER = "88572"
NUM_RUNS = 1
MAX_NEW_TOKENS = 8192
DO_SAMPLE = False  # set True for stochastic sampling
TEMPERATURE = 0.7
TOP_P = 0.95
ENABLE_THINKING = True
INJECTED_THINKING = ""  # Optional: prefill assistant <think> with this text


MODEL_NAME = "Qwen/Qwen3-14B"


def get_cache_dir():
    preferred_cache_dirs = [
        os.environ.get("HF_HOME"),
        "/runpod-volume/hf-cache" if os.path.isdir("/runpod-volume") else None,
        "/workspace/.cache/huggingface",
    ]
    cache_dir = next((p for p in preferred_cache_dirs if p), "/workspace/.cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    return cache_dir


def load_model_and_tokenizer():
    cache_dir = get_cache_dir()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="auto",
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    # Disable EOS-based stopping to avoid early termination; rely on max_new_tokens
    try:
        gen = model.generation_config
        gen.forced_eos_token_id = None
        gen.eos_token_id = None
        if getattr(gen, "pad_token_id", None) is None:
            gen.pad_token_id = (
                getattr(tokenizer, "pad_token_id", None)
                or getattr(tokenizer, "eos_token_id", None)
                or 0
            )
    except Exception:
        pass
    return tokenizer, model


def build_inputs(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    if INJECTED_THINKING:
        if "<think>" in text:
            text = text.replace("<think>", "<think>" + INJECTED_THINKING + " ", 1)
        else:
            text = text + "<think>" + INJECTED_THINKING + " "
    return tokenizer([text], return_tensors="pt")


def split_thinking_output(tokenizer, output_ids):
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return thinking, content


def split_injected_and_remainder(thinking: str, injected: str):
    if not injected:
        return "", thinking
    # Remove the injected prefix if present at the start (after optional leading whitespace)
    leading_stripped = thinking.lstrip()
    leading_ws_len = len(thinking) - len(leading_stripped)
    if leading_stripped.startswith(injected):
        remainder = leading_stripped[len(injected):]
        # If we stripped leading whitespace, keep the exact whitespace removed at the front
        return injected, remainder.lstrip()
    # If not a clean prefix, still return both for transparency
    return injected, thinking


def ensure_output_dir():
    out_dir = os.path.join(os.path.dirname(__file__), "output tables")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def timestamped_csv_path():
    out_dir = ensure_output_dir()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"run_{ts}.csv")


def _print_progress(current: int, total: int, width: int = 40):
    current = max(0, min(current, total))
    if total <= 0:
        bar = "#" * width
        suffix = "0/0"
    else:
        filled = int(math.floor(width * (current / total)))
        bar = "#" * filled + "-" * (width - filled)
        suffix = f"{current}/{total}"
    print(f"\rProgress: [{bar}] {suffix}", end="", flush=True)


def main():
    tokenizer, model = load_model_and_tokenizer()
    csv_path = timestamped_csv_path()

    rows = []
    matches = 0

    # Stop at <|im_end|> to terminate the assistant turn cleanly
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end_id = getattr(tokenizer, "eos_token_id", None)

    _print_progress(0, NUM_RUNS)
    for i in range(NUM_RUNS):
        model_inputs = build_inputs(tokenizer, PROMPT).to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE if DO_SAMPLE else None,
            top_p=TOP_P if DO_SAMPLE else None,
            eos_token_id=im_end_id,
            forced_eos_token_id=None,
            stopping_criteria=StoppingCriteriaList([]),
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        thinking, content = split_thinking_output(tokenizer, output_ids)
        injected_part, remainder_thinking = split_injected_and_remainder(thinking, INJECTED_THINKING)
        include_flag = 1 if INTENDED_ANSWER in content else 0

        rows.append({
            "model_name": MODEL_NAME,
            "prompt": PROMPT,
            "intended_answer": INTENDED_ANSWER,
            "includes_answer": include_flag,
            "injected_thinking": INJECTED_THINKING,
            "remainder_thinking": remainder_thinking,
        })

        if include_flag:
            matches += 1

        _print_progress(i + 1, NUM_RUNS)

    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "prompt",
                "intended_answer",
                "includes_answer",
                "injected_thinking",
                "remainder_thinking",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()  # newline after progress bar
    print(f"Contains matches: {matches}/{NUM_RUNS}")
    print(f"Saved rows to: {csv_path}")


if __name__ == "__main__":
    main()


