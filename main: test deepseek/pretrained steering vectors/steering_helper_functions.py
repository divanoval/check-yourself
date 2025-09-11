### Mostly borrower from public colab (https://colab.research.google.com/drive/1CXadiO7XZP216QvIyUUfhnJzKgz-EMew#scrollTo=eJ9brH3GMeDI) ###

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from nnsight import LanguageModel
import json
import os
from typing import Dict, List, Optional

# Force caches to RAM (tmpfs) to avoid filling container disk
def _configure_ram_caches():
    try:
        ram_root = "/dev/shm"
        use_ram = os.path.isdir(ram_root) and os.access(ram_root, os.W_OK)
        if use_ram:
            hf_home = os.path.join(ram_root, "hf-cache")
        else:
            # Fallback to workspace-mounted storage, still avoids container overlay
            hf_home = "/workspace/.cache/huggingface"
        os.makedirs(os.path.join(hf_home, "hub"), exist_ok=True)
        os.makedirs(os.path.join(hf_home, "transformers"), exist_ok=True)
        os.makedirs(os.path.join(hf_home, "datasets"), exist_ok=True)
        os.makedirs(os.path.join(hf_home, "torch"), exist_ok=True)

        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
        os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
        os.environ.setdefault("TORCH_HOME", os.path.join(hf_home, "torch"))

        # Ensure temp files also go to RAM when available
        if use_ram:
            os.environ.setdefault("TMPDIR", ram_root)
    except Exception:
        pass

_configure_ram_caches()

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
VECTOR_PATH = "/workspace/check-yourself/main: test deepseek/pretrained steering vectors/mean_vectors_deepseek-r1-distill-qwen-14b.pt"

STEERING_CONFIG = {
    "backtracking": {
        "vector_layer": 17,
        "pos_layers": [17],
        "neg_layers": [17],
    },
    "uncertainty-estimation": {
        "vector_layer": 18,
        "pos_layers": [18],
        "neg_layers": [18],
    },
    "example-testing": {
        "vector_layer": 15,
        "pos_layers": [15],
        "neg_layers": [15],
    },
    "adding-knowledge": {
        "vector_layer": 18,
        "pos_layers": [18],
        "neg_layers": [18],
    }
}

def load_model_and_vectors():
    """Load the model, tokenizer, and pre-trained steering vectors."""
    print(f"Loading model {MODEL_NAME}...")

    # Load model with nnsight
    model = LanguageModel(
        MODEL_NAME,
        dispatch=True,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )

    # Configure generation settings
    model.generation_config.temperature = 0.6
    model.generation_config.top_p = 0.95
    model.generation_config.do_sample = True

    tokenizer = model.tokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load pre-trained vectors
    print(f"Loading steering vectors from {VECTOR_PATH}...")
    if os.path.exists(VECTOR_PATH):
        mean_vectors_dict = torch.load(VECTOR_PATH)

        # Compute feature vectors by subtracting overall mean
        feature_vectors = {}
        feature_vectors["overall"] = mean_vectors_dict["overall"]['mean']

        for label in ["initializing", "deduction", "adding-knowledge", "example-testing", "uncertainty-estimation", "backtracking"]:
            if label != 'overall':
                feature_vectors[label] = mean_vectors_dict[label]['mean'] - mean_vectors_dict["overall"]['mean']

                # Normalize feature vectors
                for layer in range(model.config.num_hidden_layers):
                    feature_vectors[label][layer] = feature_vectors[label][layer] * (
                        feature_vectors["overall"][layer].norm() / feature_vectors[label][layer].norm()
                    )

        print("Successfully loaded steering vectors!")
        return model, tokenizer, feature_vectors
    else:
        print(f"Vector file not found at {VECTOR_PATH}")
        return model, tokenizer, {}

def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2**14,
    steering_label: Optional[str] = None, # "backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"
    feature_vectors: Optional[Dict] = None, # won't use this
    steer_positive: bool = True, 
    coefficient: float = 1.0
):
    """
    Generate text with optional steering vector intervention.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        steering_label: Which reasoning pattern to steer towards/away from
        feature_vectors: Pre-trained feature vectors
        steer_positive: If True, steer towards the pattern; if False, steer away
        coefficient: Strength of the steering intervention
    """
    # Format prompt for chat
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    # Use tracer context; apply interventions directly without calling `.all()`
    with model.generate(
        {
            "input_ids": input_ids,
            "attention_mask": (input_ids != tokenizer.pad_token_id).long()
        },
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    ) as tracer:
        # No tracer.all(); directly intervene at target layers

        # Apply steering vector if specified
        if steering_label and feature_vectors and steering_label in feature_vectors:
            config = STEERING_CONFIG[steering_label]
            vector_layer = config["vector_layer"]
            pos_layers = config["pos_layers"]
            neg_layers = config["neg_layers"]

            # Get the feature vector
            feature_vector = feature_vectors[steering_label][vector_layer].to("cuda").to(torch.bfloat16)

            # Apply steering intervention
            if steer_positive:
                for layer_idx in pos_layers:
                    hs = model.model.layers[layer_idx].output[0]
                    delta = (coefficient * feature_vector).to(hs.dtype).to(hs.device)
                    if hs.dim() == 3:
                        hs[:, :, :] += delta.view(1, 1, -1)
                    else:
                        hs[:, :] += delta.view(1, -1)
            else:
                for layer_idx in neg_layers:
                    hs = model.model.layers[layer_idx].output[0]
                    delta = (coefficient * feature_vector).to(hs.dtype).to(hs.device)
                    if hs.dim() == 3:
                        hs[:, :, :] -= delta.view(1, 1, -1)
                    else:
                        hs[:, :] -= delta.view(1, -1)

        outputs = model.generator.output.save()

    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


def generate_with_steering_from_inputs(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 2**14,
    steering_label: Optional[str] = None,
    feature_vectors: Optional[Dict] = None,
    steer_positive: bool = True,
    coefficient: float = 1.0,
):
    """
    Generate text using nnsight with optional steering, taking prebuilt input_ids/attention_mask.

    Returns the decoded generated text (not including the prompt), with special tokens preserved
    so downstream code can split on tags like <think>.
    """
    device = "cuda"
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Use tracer context; apply interventions directly without calling `.all()`
    with model.generate(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    ) as tracer:
        # No tracer.all(); directly intervene at target layers

        if steering_label and feature_vectors and steering_label in feature_vectors:
            config = STEERING_CONFIG[steering_label]
            vector_layer = config["vector_layer"]
            pos_layers = config["pos_layers"]
            neg_layers = config["neg_layers"]

            feature_vector = feature_vectors[steering_label][vector_layer].to(device).to(torch.bfloat16)

            if steer_positive:
                for layer_idx in pos_layers:
                    hs = model.model.layers[layer_idx].output[0]
                    delta = (coefficient * feature_vector).to(hs.dtype).to(hs.device)
                    if hs.dim() == 3:
                        hs[:, :, :] += delta.view(1, 1, -1)
                    else:
                        hs[:, :] += delta.view(1, -1)
            else:
                for layer_idx in neg_layers:
                    hs = model.model.layers[layer_idx].output[0]
                    delta = (coefficient * feature_vector).to(hs.dtype).to(hs.device)
                    if hs.dim() == 3:
                        hs[:, :, :] -= delta.view(1, 1, -1)
                    else:
                        hs[:, :] -= delta.view(1, -1)

        outputs = model.generator.output.save()

    # Preserve special tokens to allow downstream splitting on tags
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response
