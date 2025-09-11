from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteriaList
from transformers import StoppingCriteria
import re
import csv
import datetime
import os
import math
import torch
from steering_vector_colab import load_model_and_vectors, generate_with_steering_from_inputs


# ===== User inputs =====
FILE_NAME = "test_deepseek_36084_prefix_bit-5"
PROMPT = "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$"
F_INTENDED_ANSWER = "boxed{88572}"
F_WRONG_ANSWER = "boxed{29523}"
INTENDED_ANSWER = "88572"
WRONG_ANSWER = "36084"
NUM_RUNS = 3
MAX_NEW_TOKENS = 2**14
DO_SAMPLE = True  # set True for stochastic sampling
TEMPERATURE = 0.6
TOP_P = 0.95
ENABLE_THINKING = True
WRONG_PREFIX_FULL = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3)))))))))\). Wow, that's a lot of nested parentheses! It looks like a recursive expression, where each layer is 3 times (1 plus something). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let's identify the innermost part.
Looking at the expression, the innermost parentheses is the last one: (1 + 3). Let me compute that first.
1 + 3 = 4.
Okay, so replacing that innermost part, the expression becomes: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(4)))))))))\).
Wait, that's a bit confusing. Maybe I should label each layer as I go. Let me number them from the innermost as Layer 1 to the outermost as Layer 9.
Layer 1: (1 + 3) = 4.
Then Layer 2: 3(1 + Layer 1) = 3*(1 + 4) = 3*5 = 15.
Wait, hold on. Let me make sure. So each layer is 3*(1 + previous layer). So starting from the inside:
Layer 1: (1 + 3) = 4.
Layer 2: 3*(1 + Layer 1) = 3*(1 + 4) = 3*5 = 15.
Layer 3: 3*(1 + Layer 2) = 3*(1 + 15) = 3*16 = 48.
Layer 4: 3*(1 + Layer 3) = 3*(1 + 48) = 3*49 = 147.
Layer 5: 3*(1 + Layer 4) = 3*(1 + 147) = 3*148 = 444.
Layer 6: 3*(1 + Layer 5) = 3*(1 + 444) = 3*445 = 1335.
Layer 7: 3*(1 + Layer 6) = 3*(1 + 1335) = 3*1336 = 4008.
Layer 8: 3*(1 + Layer 7) = 3*(1 + 4008) = 3*4009 = 12027.
Layer 9: 3*(1 + Layer 8) = 3*(1 + 12027) = 3*12028 = 36084."""

WRONG_PREFIX_BIT = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3)))))))))\). Wow, that's a lot of nested parentheses! It looks like a recursive expression, where each layer is 3 times (1 plus something). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let's identify the innermost part.
Looking at the expression, the innermost parentheses is the last one: (1 + 3). Let me compute that first.
1 + 3 = 4.
Okay, so replacing that innermost part, the expression becomes: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(4)))))))))\).
Wait, that's a bit confusing. Maybe I should label each layer as I go. Let me number them from the innermost as Layer 1 to the outermost as Layer 9.
Layer 1: (1 + 3) = 4.
Then Layer 2: 3(1 + Layer 1) = 3*(1 + 4) = 3*5 = 15."""

WRONG_PREFIX_BIT_MINUS_1 = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3)))))))))\). Wow, that's a lot of nested parentheses! It looks like a recursive expression, where each layer is 3 times (1 plus something). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let's identify the innermost part.
Looking at the expression, the innermost parentheses is the last one: (1 + 3). Let me compute that first.
1 + 3 = 4.
Okay, so replacing that innermost part, the expression becomes: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(4)))))))))\).
Wait, that's a bit confusing. Maybe I should label each layer as I go. Let me number them from the innermost as Layer 1 to the outermost as Layer 9.
Layer 1: (1 + 3) = 4."""

WRONG_PREFIX_BIT_MINUS_5 = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3)))))))))\). Wow, that's a lot of nested parentheses! It looks like a recursive expression, where each layer is 3 times (1 plus something). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let's identify the innermost part.
Looking at the expression, the innermost parentheses is the last one: (1 + 3). Let me compute that first.
1 + 3 = 4.
Okay, so replacing that innermost part, the expression becomes: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(4)))))))))\)."""

WRONG_PREFIX_BIT_MINUS_9 = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3)))))))))\). Wow, that's a lot of nested parentheses! It looks like a recursive expression, where each layer is 3 times (1 plus something). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let's identify the innermost part."""

WRONG_BLUNT = "The answer is 36084."



RIGHT_PREFIX_FULL = """Okay, so I need to compute this big expression..."""


INJECTED_THINKING = WRONG_PREFIX_BIT_MINUS_5


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


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


def _normalize_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    return prefix.rstrip("\n") + "\n"


def build_inputs(tokenizer, prompt: str, injected_prefix: str, injected_thinking: str):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    # Inject prefix (with exactly one trailing newline), and optionally injected_thinking immediately after it
    insert_after_think = _normalize_prefix(injected_prefix) + (injected_thinking + " " if injected_thinking else "")
    if insert_after_think:
        if "<think>" in text:
            text = text.replace("<think>", "<think>" + insert_after_think, 1)
        else:
            # Model did not emit <think>; pre-emit <think> + prefix (and thinking if provided)
            text = text + "<think>" + insert_after_think
    return tokenizer([text], return_tensors="pt")


def split_thinking_output(tokenizer, output_ids):
    # Decode with special tokens kept so we can string-split reliably
    raw = tokenizer.decode(output_ids, skip_special_tokens=False)

    # Known variants across model families
    START_CANDIDATES = ["<think>", "<|assistant_think|>", "<|begin_think|>"]
    END_CANDIDATES   = ["</think>", "<|end_think|>"]

    start = -1
    start_tag = None
    for t in START_CANDIDATES:
        pos = raw.find(t)
        if pos != -1:
            start, start_tag = pos, t
            break

    end = -1
    end_tag = None
    if start != -1:
        for t in END_CANDIDATES:
            pos = raw.find(t, start + len(start_tag))
            if pos != -1:
                end, end_tag = pos, t
                break

    if start != -1 and end != -1 and end > start:
        thinking = raw[start + len(start_tag): end].strip()
        content  = raw[end + len(end_tag):].strip()
        return thinking, tokenizer.decode(tokenizer.encode(content), skip_special_tokens=True)

    # Fallbacks: treat everything as content if we can't find a proper block
    return "", tokenizer.decode(output_ids, skip_special_tokens=True)


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


def timestamped_csv_path(file_name: str):
    out_dir = ensure_output_dir()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{file_name}.csv")


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


def run_evaluation(file_name: str,
                   prompt: str,
                   f_intended_answer: str,
                   intended_answer: str,
                   num_runs: int,
                   injected_thinking: str,
                   injection_depth: int = 0,
                   injected_prefix: str | None = None):
    tokenizer, model = load_model_and_tokenizer()
    csv_path = timestamped_csv_path(file_name)

    rows = []
    matches = 0
    thinking_intended_matches = 0
    sum_token_count = 0
    produced_answers = []  # ordered list of distinct model_answer values
    _seen_answers = {}

    # Stop at <|im_end|> to terminate the assistant turn cleanly
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end_id = getattr(tokenizer, "eos_token_id", None)

    _print_progress(0, num_runs)
    for i in range(num_runs):
        # Stop when boxed{...} appears
        class StopOnBoxed(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                return re.search(r"boxed\{[^}]+\}", text) is not None

        injection_occurred = False
        phase1_out_ids: list[int] = []
        phase2_out_ids: list[int] = []

        if injection_depth and injection_depth > 0:
            # Phase 1: generate up to injection_depth with StopOnBoxed enabled and no injected thinking
            model_inputs_phase1 = build_inputs(tokenizer, prompt, injected_prefix or "", "").to(model.device)
            generated_ids_phase1 = model.generate(
                **model_inputs_phase1,
                max_new_tokens=injection_depth,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE if DO_SAMPLE else None,
                top_p=TOP_P if DO_SAMPLE else None,
                eos_token_id=im_end_id,
                forced_eos_token_id=None,
                stopping_criteria=StoppingCriteriaList([StopOnBoxed()]),
            )
            phase1_out_ids = generated_ids_phase1[0][len(model_inputs_phase1.input_ids[0]):].tolist()

            # If boxed appeared early (< injection_depth), skip injection entirely
            if len(phase1_out_ids) < injection_depth:
                output_ids = phase1_out_ids
            else:
                # Inject thinking text without counting as generated tokens
                injection_occurred = True
                injected_ids = tokenizer.encode(injected_thinking, add_special_tokens=False)

                # Build new prefix: original prompt + phase1_out + injected
                prompt_ids = model_inputs_phase1.input_ids[0].tolist()
                new_prefix_ids = prompt_ids + phase1_out_ids + injected_ids
                new_input_ids = torch.tensor([new_prefix_ids], dtype=torch.long, device=model.device)
                new_attention_mask = torch.ones_like(new_input_ids)

                generated_ids_phase2 = model.generate(
                    input_ids=new_input_ids,
                    attention_mask=new_attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE if DO_SAMPLE else None,
                    top_p=TOP_P if DO_SAMPLE else None,
                    eos_token_id=im_end_id,
                    forced_eos_token_id=None,
                    stopping_criteria=StoppingCriteriaList([StopOnBoxed()]),
                )
                # Extract only tokens generated after the new prefix
                phase2_out_ids = generated_ids_phase2[0][new_input_ids.shape[1]:].tolist()
                output_ids = phase1_out_ids + phase2_out_ids
        else:
            # Single-pass generation as before
            model_inputs = build_inputs(tokenizer, prompt, injected_prefix or "", injected_thinking).to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE if DO_SAMPLE else None,
                top_p=TOP_P if DO_SAMPLE else None,
                eos_token_id=im_end_id,
                forced_eos_token_id=None,
                stopping_criteria=StoppingCriteriaList([StopOnBoxed()]),
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        token_count = len(output_ids)
        sum_token_count += token_count
        thinking, content = split_thinking_output(tokenizer, output_ids)
        injected_part, remainder_thinking = split_injected_and_remainder(thinking, injected_thinking)
        include_flag = 1 if (f_intended_answer in content or f_intended_answer in remainder_thinking) else 0
        wrong_include_flag = 1 if (F_WRONG_ANSWER in content or F_WRONG_ANSWER in remainder_thinking) else 0
        include_intended_in_thinking = remainder_thinking.count(intended_answer) + content.count(intended_answer)
        include_wrong_in_thinking = remainder_thinking.count(WRONG_ANSWER) + content.count(WRONG_ANSWER)

        # Extract the last occurrence of boxed{...} from the combined model output
        combined_output_for_search = f"{remainder_thinking}\n{content}"
        boxed_matches = re.findall(r"boxed\{(.*?)\}", combined_output_for_search, flags=re.DOTALL)
        model_answer = boxed_matches[-1] if boxed_matches else ""
        if model_answer:
            if model_answer not in _seen_answers:
                _seen_answers[model_answer] = 1
            else:
                _seen_answers[model_answer] += 1

        # Ensure single-line fields for CSV
        content_one_line = re.sub(r"[\r\n]+", " <nEwLiNe>", content).strip()
        remainder_one_line = re.sub(r"[\r\n]+", " <nEwLiNe>", remainder_thinking).strip()
        injected_one_line = re.sub(r"[\r\n]+", " <nEwLiNe>", injected_thinking).strip()

        # Single CSV row with a multi-line output cell
        if injection_occurred:
            # Build phase-specific outputs to place the tag between them
            # Phase 1 text
            p1_think, p1_cont = split_thinking_output(tokenizer, phase1_out_ids) if phase1_out_ids else ("", "")
            _, p1_remainder = split_injected_and_remainder(p1_think, "")
            p1_out = (re.sub(r"[\r\n]+", " <nEwLiNe>", p1_remainder).strip() + " " +
                      re.sub(r"[\r\n]+", " <nEwLiNe>", p1_cont).strip()).strip()

            # Phase 2 text
            p2_think, p2_cont = split_thinking_output(tokenizer, phase2_out_ids) if phase2_out_ids else ("", "")
            _, p2_remainder = split_injected_and_remainder(p2_think, "")
            p2_out = (re.sub(r"[\r\n]+", " <nEwLiNe>", p2_remainder).strip() + " " +
                      re.sub(r"[\r\n]+", " <nEwLiNe>", p2_cont).strip()).strip()

            combined_out = p1_out + (" <<INJECTED THINKING>> " if p2_out else " <<INJECTED THINKING>> ") + p2_out
            output_multiline = "\n" + combined_out
        else:
            output_multiline = "\n" + (remainder_one_line + " " + content_one_line)
        rows.append({
            "index": i + 1,
            "model_answer": model_answer,
            "correct": include_flag,
            "includes_intended": include_intended_in_thinking,
            "token_count": token_count,
            "output": output_multiline,
        })

        if include_flag:
            matches += 1
        if include_intended_in_thinking:
            thinking_intended_matches += 1

        _print_progress(i + 1, num_runs)

    produced_answers = [f"{a} ({c})" for a, c in _seen_answers.items() if a.strip()]
    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        # Write summary stats at the top of the CSV
        avg_token_count = (sum_token_count / num_runs) if num_runs else 0
        f.write(f"Answers produced: {', '.join(produced_answers)}\n")
        f.write(f"Injection depth (tokens): {injection_depth}\n")
        f.write(f"Model name: {MODEL_NAME}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Intended answer: {intended_answer}\n\n")
        if injected_prefix:
            f.write(f"Injected prefix: \n {injected_prefix}\n\n")
        f.write(f"Injected thinking: \n {injected_thinking}\n\n")
        f.write(f"Average token_count: {avg_token_count:.2f}\n")
        f.write(f"Correct answers: {matches}/{num_runs}\n")
        f.write(f"Output contains correct answer: {thinking_intended_matches}/{num_runs}\n\n")

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "model_answer",
                "correct",
                "includes_intended",
                "token_count",
                "output"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()  # newline after progress bar
    print(f"Correct answers: {matches}/{num_runs}")
    print(f"Output contains correct answer: {thinking_intended_matches}/{num_runs}")
    print(f"Answers produced: {', '.join(produced_answers)}")
    print(f"Saved rows to: {csv_path}")


if __name__ == "__main__":
    run_evaluation(
        FILE_NAME,
        PROMPT,
        F_INTENDED_ANSWER,
        INTENDED_ANSWER,
        NUM_RUNS,
        INJECTED_THINKING,
        injected_prefix=None,
    )


def run_evaluation_with_steering(file_name: str,
                                 prompt: str,
                                 f_intended_answer: str,
                                 intended_answer: str,
                                 num_runs: int,
                                 injected_prefix: str | None,
                                 steering_label: str | None,
                                 steer_positive: bool = True,
                                 coefficient: float = 1.0):
    # Load nnsight model/tokenizer and steering vectors
    steer_model, steer_tokenizer, feature_vectors = load_model_and_vectors()
    csv_path = timestamped_csv_path(file_name)

    rows = []
    matches = 0
    thinking_intended_matches = 0
    sum_token_count = 0
    produced_answers = []
    _seen_answers = {}

    # Build initial input once (prompt + injected_prefix at <think>)
    messages = [{"role": "user", "content": prompt}]
    text = steer_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    insert_after_think = _normalize_prefix(injected_prefix or "")
    if insert_after_think:
        if "<think>" in text:
            text = text.replace("<think>", "<think>" + insert_after_think, 1)
        else:
            text = text + "<think>" + insert_after_think

    base_inputs = steer_tokenizer([text], return_tensors="pt")

    # Stop by post-parsing; nnsight does not support transformers StoppingCriteria directly
    _print_progress(0, num_runs)
    for i in range(num_runs):
        input_ids = base_inputs.input_ids.to("cuda")
        attention_mask = (input_ids != steer_tokenizer.pad_token_id).long()

        # Generate steered output (single pass)
        response_text = generate_with_steering_from_inputs(
            steer_model,
            steer_tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            steering_label=steering_label,
            feature_vectors=feature_vectors,
            steer_positive=steer_positive,
            coefficient=coefficient,
        )

        # Decode section after the prompt by subtracting the prompt string prefix
        # We rely on tokenizer to reconstruct tags; response_text includes generated text.
        full_raw = response_text

        # Extract thinking/content like split_thinking_output expects token IDs; reuse text splitter
        # Using a textual variant: emulate split_thinking_output by searching tags in full_raw
        START_CANDIDATES = ["<think>", "<|assistant_think|>", "<|begin_think|>"]
        END_CANDIDATES = ["</think>", "<|end_think|>"]
        start = -1
        start_tag = None
        for t in START_CANDIDATES:
            pos = full_raw.find(t)
            if pos != -1:
                start, start_tag = pos, t
                break
        end = -1
        end_tag = None
        if start != -1:
            for t in END_CANDIDATES:
                pos = full_raw.find(t, start + len(start_tag))
                if pos != -1:
                    end, end_tag = pos, t
                    break
        if start != -1 and end != -1 and end > start:
            thinking = full_raw[start + len(start_tag): end].strip()
            content = full_raw[end + len(end_tag):].strip()
        else:
            thinking = ""
            content = full_raw

        injected_part, remainder_thinking = split_injected_and_remainder(thinking, (injected_prefix or "").rstrip("\n"))

        include_flag = 1 if (f_intended_answer in content or f_intended_answer in remainder_thinking) else 0
        include_intended_in_thinking = remainder_thinking.count(intended_answer) + content.count(intended_answer)

        # Extract last boxed{...}
        combined_output_for_search = f"{remainder_thinking}\n{content}"
        boxed_matches = re.findall(r"boxed\{(.*?)\}", combined_output_for_search, flags=re.DOTALL)
        model_answer = boxed_matches[-1] if boxed_matches else ""
        if model_answer:
            _seen_answers[model_answer] = _seen_answers.get(model_answer, 0) + 1

        content_one_line = re.sub(r"[\r\n]+", " <nEwLiNe>", content).strip()
        remainder_one_line = re.sub(r"[\r\n]+", " <nEwLiNe>", remainder_thinking).strip()

        output_multiline = "\n" + (remainder_one_line + " " + content_one_line)
        rows.append({
            "index": i + 1,
            "model_answer": model_answer,
            "correct": include_flag,
            "includes_intended": include_intended_in_thinking,
            "token_count": len(steer_tokenizer.encode(content, add_special_tokens=False)),
            "output": output_multiline,
        })

        if include_flag:
            matches += 1
        if include_intended_in_thinking:
            thinking_intended_matches += 1

        sum_token_count += rows[-1]["token_count"]
        _print_progress(i + 1, num_runs)

    produced_answers = [f"{a} ({c})" for a, c in _seen_answers.items() if a.strip()]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        avg_token_count = (sum_token_count / num_runs) if num_runs else 0
        f.write(f"Answers produced: {', '.join(produced_answers)}\n")
        f.write(f"Model name: {MODEL_NAME}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Intended answer: {intended_answer}\n\n")
        if injected_prefix:
            f.write(f"Injected prefix: \n {injected_prefix}\n\n")
        f.write(f"Steering label: {steering_label}\n")
        f.write(f"Steer positive: {steer_positive}\n")
        f.write(f"Coefficient: {coefficient}\n")
        f.write(f"Average token_count: {avg_token_count:.2f}\n")
        f.write(f"Correct answers: {matches}/{num_runs}\n")
        f.write(f"Output contains correct answer: {thinking_intended_matches}/{num_runs}\n\n")

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "model_answer",
                "correct",
                "includes_intended",
                "token_count",
                "output"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Correct answers: {matches}/{num_runs}")
    print(f"Output contains correct answer: {thinking_intended_matches}/{num_runs}")
    print(f"Answers produced: {', '.join(produced_answers)}")
    print(f"Saved rows to: {csv_path}")


