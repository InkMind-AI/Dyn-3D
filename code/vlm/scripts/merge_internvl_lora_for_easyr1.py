#!/usr/bin/env python3
"""Merge an InternVL LoRA adapter into a standalone EasyR1/vLLM model."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    # EasyR1 adapters generally contain only LoRA tensors and may not carry a
    # tokenizer.  The tokenizer belongs to the SFT base model and must be
    # preserved in the standalone merged artifact.
    AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(output)

    base = Path(args.base_model)
    for pattern in ("configuration_*.py", "modeling_*.py", "conversation.py", "chat_template.jinja"):
        for source in base.glob(pattern):
            shutil.copy2(source, output / source.name)


if __name__ == "__main__":
    main()
