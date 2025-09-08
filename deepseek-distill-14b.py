from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
import re
import os
import sys


model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
THINK_END_ID = 151668

# ensure model/tokenizer cache is on the pod (e.g., Runpod volume)
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


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        cache_dir=cache_dir,
        low_cpu_mem_usage=True
    )
    # Disable EOS-based stopping in config to avoid early termination
    try:
        gen = model.generation_config
        gen.forced_eos_token_id = None
        gen.eos_token_id = None
    except Exception:
        pass
    return tokenizer, model


def _im_end_id(tokenizer):
    try:
        tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(tid, int) and tid >= 0:
            return tid
    except Exception:
        pass
    return getattr(tokenizer, "eos_token_id", None)


class StopAtImEndAfterThinkEnd(StoppingCriteria):
    """Stop only when <|im_end|> appears AFTER </think> has been generated."""

    def __init__(self, im_end_id: int, think_end_id: int):
        super().__init__()
        self.im_end_id = im_end_id
        self.think_end_id = think_end_id
        self.seen_think_end = False

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0].tolist()
        if not self.seen_think_end and self.think_end_id in seq:
            self.seen_think_end = True
        if self.seen_think_end and len(seq) > 0 and seq[-1] == self.im_end_id:
            return True
        return False


class StopOnBoxed(StoppingCriteria):
    """Stop when a pattern boxed{...} appears in the generated text."""

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return re.search(r"boxed\{[^}]+\}", text) is not None


def run_inference(tokenizer, model, prompt: str, max_new_tokens: int = 2**16, enable_thinking: bool = True, injected_thinking: str = None):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    if injected_thinking:
        if "<think>" in text:
            text = text.replace("<think>", "<think>" + injected_thinking + " ", 1)
        else:
            text = text + "<think>" + injected_thinking + " "
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    eot_id = _im_end_id(tokenizer)
    stopping = StopAtImEndAfterThinkEnd(eot_id, THINK_END_ID)
    stop_boxed = StopOnBoxed(tokenizer)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=None,
        forced_eos_token_id=None,
        stopping_criteria=StoppingCriteriaList([stop_boxed, stopping])
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_END_ID)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return thinking_content, content


class ThinkingColorStreamer(TextStreamer):
    """Stream tokens as they are generated; colorize thinking and split at </think>.

    Uses token id 151668 as end-of-thinking sentinel when present.
    """

    def __init__(self, tokenizer, color_thinking: str = "\033[35m", color_reset: str = "\033[0m"):
        super().__init__(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=False)
        self.color_thinking = color_thinking
        self.color_reset = color_reset
        self.has_switched = False
        try:
            self._think_end_str = tokenizer.decode([151668], skip_special_tokens=False)
        except Exception:
            self._think_end_str = "</think>"

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if not self.has_switched:
            if self._think_end_str and self._think_end_str in text:
                before, _, after = text.partition(self._think_end_str)
                if before:
                    print(self.color_thinking + before + self.color_reset, end="", flush=True)
                print("\ncontent: ", end="", flush=True)
                self.has_switched = True
                if after:
                    print(after, end="", flush=True)
            else:
                print(self.color_thinking + text + self.color_reset, end="", flush=True)
        else:
            print(text, end="", flush=True)
        if stream_end:
            print("", end="\n", flush=True)


def run_inference_streaming(tokenizer, model, prompt: str, max_new_tokens: int = 2**16, enable_thinking: bool = True, injected_thinking: str = None):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    if injected_thinking:
        if "<think>" in text:
            text = text.replace("<think>", "<think>" + injected_thinking + " ", 1)
        else:
            text = text + "<think>" + injected_thinking + " "
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = ThinkingColorStreamer(tokenizer)
    print("thinking content: ", end="", flush=True)
    eot_id = _im_end_id(tokenizer)
    stopping = StopAtImEndAfterThinkEnd(eot_id, THINK_END_ID)
    stop_boxed = StopOnBoxed(tokenizer)
    _ = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        eos_token_id=None,
        forced_eos_token_id=None,
        stopping_criteria=StoppingCriteriaList([stop_boxed, stopping])
    )


def read_multiline_input(invite: str, end_marker: str = "<<<END>>>"):
    print(f"{invite} — paste multiple lines, end with {end_marker}. Press Enter to skip.", flush=True)
    lines = []
    first = True
    while True:
        try:
            line = input()
        except EOFError:
            break
        if first and line.strip() == "":
            return None
        first = False
        if line.strip() == end_marker:
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    return text if text else None


if __name__ == "__main__":
    tokenizer, model = load_model_and_tokenizer()

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        run_inference_streaming(tokenizer, model, prompt, max_new_tokens=2**16, enable_thinking=True)
        sys.exit(0)

    print("Interactive mode. Type 'exit' or press Ctrl-D to quit.")
    try:
        while True:
            try:
                prompt = input("> ").strip()
            except EOFError:
                break
            if not prompt:
                continue
            if prompt.lower() in {"exit", "quit", "q"}:
                break

            injected = read_multiline_input("(optional) injected thinking")

            run_inference_streaming(tokenizer, model, prompt, max_new_tokens=2**16, enable_thinking=True, injected_thinking=injected)
    finally:
        pass

