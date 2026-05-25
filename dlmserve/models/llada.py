"""LLaDA-8B and LLaDA-1.5 loader.

Loads model and tokenizer from HuggingFace. Both variants share this loader;
dispatch is by model ID prefix in `engine.load_model()`.

Measured VRAM (RTX 5070 12 GB, INT4): ~5.59 GB weights, ~5.90 GB peak during
forward pass. BF16 is ~16 GB and does not fit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "gsai-ml/LLaDA-8B-Instruct"

MASK_TOKEN = "<|mdm_mask|>"
MASK_ID = 126336
BASE_VOCAB_SIZE = 126080
OUTPUT_VOCAB_SIZE = 126464

Dtype = Literal["int4", "bf16"]


@dataclass
class LoadedLLaDA:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: Dtype
    mask_id: int = MASK_ID


def load_llada(
    model_id: str = DEFAULT_MODEL_ID,
    dtype: Dtype = "int4",
    device: str | torch.device = "cuda",
) -> LoadedLLaDA:
    """Load LLaDA + tokenizer at the requested precision."""
    log.info("loading LLaDA tokenizer", extra={"model_id": model_id})
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id == MASK_ID:
        raise RuntimeError(
            f"pad_token_id ({tokenizer.pad_token_id}) equals mask id; "
            f"the LLaDA reference generate() does not support this."
        )

    mdm_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    if mdm_id != MASK_ID:
        raise RuntimeError(f"{MASK_TOKEN!r} resolved to {mdm_id}, expected {MASK_ID}")

    if dtype == "int4":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModel.from_pretrained(
            model_id,
            quantization_config=qconfig,
            device_map={"": device} if isinstance(device, str) else None,
            trust_remote_code=True,
        )
    elif dtype == "bf16":
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
    else:
        raise ValueError(f"unsupported dtype: {dtype!r}")

    model.eval()
    resolved_device = next(model.parameters()).device
    elapsed = time.monotonic() - t0
    log.info(
        "LLaDA loaded",
        extra={"model_id": model_id, "elapsed_ms": round(elapsed * 1000)},
    )
    return LoadedLLaDA(model=model, tokenizer=tokenizer, device=resolved_device, dtype=dtype)
