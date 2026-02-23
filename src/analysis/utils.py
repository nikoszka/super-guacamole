"""Shared utilities for analysis scripts."""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from models.huggingface_models import get_hf_cache_dir

logger = logging.getLogger(__name__)


def load_tokenizer(model_name: str):
    """Load the tokenizer for a model, supporting Llama, Qwen, and Mistral families.

    Strips quantization suffixes (-8bit, -4bit) before resolving the HuggingFace
    model ID so that the same function works for both full-precision and quantized
    model names recorded in experiment configs.
    """
    cache_dir = get_hf_cache_dir()
    name_lower = model_name.lower()

    # Strip quantization suffixes for tokenizer lookup
    base_name = model_name
    if base_name.endswith("-8bit"):
        base_name = base_name[: -len("-8bit")]
    elif base_name.endswith("-4bit"):
        base_name = base_name[: -len("-4bit")]

    if "llama" in name_lower:
        if any(tag in model_name for tag in ("Llama-3", "Llama-2", "Meta-Llama")):
            org = "meta-llama"
        else:
            org = "huggyllama"
        model_id = f"{org}/{base_name}"
        logger.info(f"Loading Llama tokenizer: {model_id}")
        return AutoTokenizer.from_pretrained(
            model_id, token_type_ids=None, cache_dir=cache_dir
        )

    if "qwen" in name_lower:
        model_id = f"Qwen/{base_name}"
        logger.info(f"Loading Qwen tokenizer: {model_id}")
        return AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir
        )

    if "mistral" in name_lower or "ministral" in name_lower:
        model_id = f"mistralai/{base_name}"
        logger.info(f"Loading Mistral tokenizer: {model_id}")
        if "ministral" in name_lower:
            return AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True,
                clean_up_tokenization_spaces=False,
                cache_dir=cache_dir,
            )
        return AutoTokenizer.from_pretrained(
            model_id,
            clean_up_tokenization_spaces=False,
            cache_dir=cache_dir,
        )

    raise ValueError(
        f"Unknown model family for '{model_name}'. "
        "Supported families: Llama, Qwen, Mistral/Ministral."
    )
