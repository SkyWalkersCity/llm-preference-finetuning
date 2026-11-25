"""Model construction: a decoder LLM with a 3-way classification head, fine-tuned with LoRA."""
from __future__ import annotations

from typing import Optional, Sequence

import torch

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
NUM_LABELS = 3


def pick_dtype(prefer: str = "auto") -> torch.dtype:
    """bf16 if the GPU supports it, else fp16 on CUDA, else fp32 (CPU/MPS tests)."""
    if prefer == "fp32" or not torch.cuda.is_available():
        return torch.float32
    if prefer == "bf16" or (prefer == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def load_base_model(model_name: str, dtype: torch.dtype, load_in_4bit: bool = False):
    """Load a sequence-classification model; optionally 4-bit quantised (QLoRA, needs bitsandbytes + CUDA)."""
    from transformers import AutoModelForSequenceClassification

    kwargs = dict(num_labels=NUM_LABELS, torch_dtype=dtype)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    return model


def set_pad_token(model, tokenizer) -> None:
    """Sequence classification pools the last non-pad token, so a pad id must be defined."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


def wrap_with_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                   target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
                   gradient_checkpointing: bool = False, quantized: bool = False):
    """Attach LoRA adapters to all attention/MLP projections and keep the new `score` head trainable.

    Trainable parameters are cast to fp32 so mixed-precision training (fp16 GradScaler) works
    even though the frozen base weights stay in half precision.
    """
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    config = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=alpha, lora_dropout=dropout,
                        target_modules=list(target_modules), bias="none")
    model = get_peft_model(model, config)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def count_parameters(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "trainable_pct": 100.0 * trainable / max(total, 1)}
