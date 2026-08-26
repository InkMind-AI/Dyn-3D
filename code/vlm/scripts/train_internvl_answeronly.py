#!/usr/bin/env python3

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for InternVL on multi-frame data.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--sample-limit", type=int, default=0)
    return parser.parse_args()


def load_rows(path: Path, sample_limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if sample_limit > 0 and len(rows) >= sample_limit:
                    break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def replace_image_tokens(text: str, image_count: int, num_image_token: int) -> str:
    replacement = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token + IMG_END_TOKEN
    expected = text.count("<image>")
    if expected != image_count:
        raise ValueError(f"Prompt has {expected} image tokens but row has {image_count} images")
    for _ in range(image_count):
        text = text.replace("<image>", replacement, 1)
    return text


class InternVLSFTDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        conv_template,
        system_message: str,
        num_image_token: int,
        input_size: int,
        max_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.system_message = system_message
        self.num_image_token = num_image_token
        self.transform = build_transform(input_size)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def _conversation_text(self, user_text: str, assistant_text: str | None) -> str:
        template = self.conv_template.copy()
        template.system_message = self.system_message
        template.append_message(template.roles[0], user_text)
        template.append_message(template.roles[1], assistant_text)
        return template.get_prompt()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        messages = row.get("messages") or []
        if len(messages) < 2:
            raise ValueError(f"Row {row.get('id', index)} does not contain user and assistant messages")
        user_text = str(messages[0].get("content", ""))
        answer_text = str(messages[1].get("content", "")).strip()
        image_paths = [str(path) for path in row.get("images") or []]
        if not image_paths:
            raise ValueError(f"Row {row.get('id', index)} has no images")

        prompt_text = self._conversation_text(user_text, None)
        full_text = self._conversation_text(user_text, answer_text)
        prompt_text = replace_image_tokens(prompt_text, len(image_paths), self.num_image_token)
        full_text = replace_image_tokens(full_text, len(image_paths), self.num_image_token)

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Conversation prompt is not a prefix for row {row.get('id', index)}")
        if len(full_ids) > self.max_length:
            raise ValueError(
                f"Row {row.get('id', index)} token length {len(full_ids)} exceeds max_length={self.max_length}"
            )

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        pixels = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                pixels.append(self.transform(image.convert("RGB")))

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": torch.stack(pixels),
            "image_flags": torch.ones(len(pixels), 1, dtype=torch.long),
        }


class InternVLCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [feature["input_ids"] for feature in features], batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = pad_sequence(
            [feature["attention_mask"] for feature in features], batch_first=True, padding_value=0
        )
        labels = pad_sequence([feature["labels"] for feature in features], batch_first=True, padding_value=-100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.cat([feature["pixel_values"] for feature in features], dim=0),
            "image_flags": torch.cat([feature["image_flags"] for feature in features], dim=0),
        }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.vision_model.requires_grad_(False)
    model.mlp1.requires_grad_(False)
    model.config.use_cache = False
    model.language_model.config.use_cache = False
    model.language_model.enable_input_require_grads()
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        model.print_trainable_parameters()

    rows = load_rows(Path(args.dataset), args.sample_limit)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    eval_count = max(1, int(round(len(rows) * args.eval_ratio))) if len(rows) > 1 else 0
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]
    base_model = model.get_base_model()
    dataset_args = (
        tokenizer,
        base_model.conv_template.copy(),
        base_model.system_message,
        base_model.num_image_token,
        args.input_size,
        args.max_length,
    )
    train_dataset = InternVLSFTDataset(train_rows, *dataset_args)
    eval_dataset = (
        InternVLSFTDataset(eval_rows, *dataset_args)
        if eval_rows
        else None
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=False,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
        optim="adamw_torch_fused",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=InternVLCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(output_dir / "final_adapter")
    if trainer.is_world_process_zero():
        manifest = {
            "base_model": str(model_path),
            "dataset": args.dataset,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "input_size": args.input_size,
            "tiles_per_frame": 1,
            "effective_batch_size": (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * max(1, training_args.world_size)
            ),
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
        }
        (output_dir / "training_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    import os

    try:
        main()
    except Exception as exc:
        print(f"InternVL SFT failed: {exc!r}", file=sys.stderr)
        raise
