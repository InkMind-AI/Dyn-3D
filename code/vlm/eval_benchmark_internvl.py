#!/usr/bin/env python3
"""Evaluate InternVL on the aligned multi-frame benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from eval_benchmark_qwen3vl import (
    extract_answer_letter,
    load_existing_results,
    resolve_frame_paths,
    resolve_sample_rows,
    shard_rows,
    summarize_results,
    write_status,
)


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-path", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--status-jsonl-path", required=True)
    parser.add_argument("--sample-size", type=int, default=999999999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-file", default="")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--answer-instruction", default="Please answer with the option letter.")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def transform(size: int) -> T.Compose:
    return T.Compose([
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def load_frames(paths: list[str], size: int) -> tuple[torch.Tensor, list[int]]:
    image_transform = transform(size)
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image_transform(image.convert("RGB")))
    return torch.stack(frames), [1] * len(frames)


def unwrap_chat_model(model):
    if isinstance(model, PeftModel):
        return model.get_base_model()
    return model


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results_sft.jsonl"
    summary_path = output_dir / "summary.json"
    latest_status = Path(args.status_path)
    status_jsonl = Path(args.status_jsonl_path)

    rows, _ = resolve_sample_rows(
        Path(args.benchmark_path), args.sample_size, args.seed, args.sample_file
    )
    rows = shard_rows(rows, args.shard_index, args.shard_count)
    existing = load_existing_results(result_path) if args.resume else []
    by_id = {str(row.get("id")): row for row in existing}
    pending = [row for row in rows if str(row.get("id")) not in by_id]
    status_base = {
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path or None,
        "output_dir": str(output_dir),
        "total_samples": len(rows),
        "completed_samples": len(existing),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    write_status(latest_status, status_jsonl, {**status_base, "state": "loading_model"})

    model = AutoModel.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True, use_fast=False
    )
    chat_model = unwrap_chat_model(model)
    device = next(chat_model.parameters()).device
    write_status(latest_status, status_jsonl, {**status_base, "state": "running"})

    mode = "a" if args.resume and result_path.exists() else "w"
    results = list(existing)
    with result_path.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(pending, start=len(existing) + 1):
            frame_paths = resolve_frame_paths(Path(args.frames_root), row)
            raw_response = ""
            predicted_answer = None
            if frame_paths:
                pixels, patches = load_frames(frame_paths, args.input_size)
                pixels = pixels.to(device=device, dtype=torch.bfloat16)
                prefix = "".join(f"Frame {i + 1}: <image>\n" for i in range(len(frame_paths)))
                prompt = prefix + str(row["question"]).strip()
                if args.answer_instruction.strip():
                    prompt += "\n\n" + args.answer_instruction.strip()
                with torch.inference_mode():
                    raw_response = chat_model.chat(
                        tokenizer,
                        pixels,
                        prompt,
                        {"max_new_tokens": args.max_new_tokens, "do_sample": False},
                        num_patches_list=patches,
                        history=None,
                        return_history=False,
                    ).strip()
                predicted_answer = extract_answer_letter(raw_response)
            gold = str(row.get("correct_answer", "")).strip().upper()
            record = {
                "id": row.get("id"),
                "scene_id": row.get("scene_id"),
                "trajectory_name": row.get("trajectory_name"),
                "type_name": row.get("type_name"),
                "question": row.get("question"),
                "correct_answer": gold,
                "predicted_answer": predicted_answer,
                "is_correct": predicted_answer == gold,
                "raw_response": raw_response,
                "frame_paths": frame_paths,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            results.append(record)
            if args.empty_cache_every and index % args.empty_cache_every == 0:
                torch.cuda.empty_cache()
            if index == 1 or index % args.progress_every == 0 or index == len(rows):
                metrics = summarize_results(results)
                write_status(latest_status, status_jsonl, {
                    **status_base,
                    "state": "running",
                    "completed_samples": index,
                    "accuracy": metrics["accuracy"],
                    "parse_rate": metrics["parse_rate"],
                })

    summary = summarize_results(results)
    summary.update({
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path or None,
        "result_path": str(result_path),
        "frames_root": args.frames_root,
        "input_size": args.input_size,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    })
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_status(latest_status, status_jsonl, {
        **status_base,
        "state": "completed",
        "completed_samples": len(rows),
        "accuracy": summary["accuracy"],
        "parse_rate": summary["parse_rate"],
        "timestamp": datetime.now().isoformat(),
    })


if __name__ == "__main__":
    main()
