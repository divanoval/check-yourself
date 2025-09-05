from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteriaList
import csv
import datetime
import os


# ===== User inputs =====
PROMPT = "Paste your prompt here"
INTENDED_ANSWER = "Paste the intended exact answer here"
NUM_RUNS = 100
MAX_NEW_TOKENS = 1024
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
        torch_dtype="auto",
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


def ensure_output_dir():
    out_dir = os.path.join(os.path.dirname(__file__), "output tables")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def timestamped_csv_path():
    out_dir = ensure_output_dir()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"run_{ts}.csv")


def main():
    tokenizer, model = load_model_and_tokenizer()
    csv_path = timestamped_csv_path()

    rows = []
    matches = 0

    for _ in range(NUM_RUNS):
        model_inputs = build_inputs(tokenizer, PROMPT).to(model.device)

        # Stop at <|im_end|> to terminate the assistant turn cleanly
        try:
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        except Exception:
            im_end_id = getattr(tokenizer, "eos_token_id", None)

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
        rows.append({"prompt": PROMPT, "thinking": thinking, "output": content})

        if content == INTENDED_ANSWER:  # strict equality
            matches += 1

    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "thinking", "output"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exact matches: {matches}/{NUM_RUNS}")
    print(f"Saved rows to: {csv_path}")


if __name__ == "__main__":
    main()


