#!/usr/bin/env python3

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from eval_vsi_mcq_qwen3vl import (
    ANSWER_INSTRUCTION,
    THREE_LINE_RE,
    build_prompt,
    extract_answer_letter,
    load_jsonl,
    load_mcq_rows,
    load_pruned_ids,
    resolve_frame_paths,
    shard_rows,
    summarize_results,
    write_status,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate InternVL on VSI-Bench MCQ with cached video frames.")
    release_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--test-jsonl", default=str(release_root / "data/vsi_bench/test.jsonl"))
    parser.add_argument("--pruned-ids", default=str(release_root / "data/vsi_bench/pruned_ids.txt"))
    parser.add_argument("--frame-cache-root", default=str(release_root / "data/vsi_bench/frame_cache"))
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--status-jsonl-path", required=True)
    parser.add_argument("--subset", choices=["full", "debiased", "pruned"], default="full")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument("--frame-dir-name", default="32f")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--answer-instruction", default=ANSWER_INSTRUCTION)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_frames(frame_paths: list[str], input_size: int) -> tuple[torch.Tensor, list[int]]:
    transform = build_transform(input_size)
    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(transform(image.convert("RGB")))
    return torch.stack(frames), [1] * len(frames)


def unwrap_chat_model(model):
    if isinstance(model, PeftModel):
        return model.get_base_model()
    return model


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_status = Path(args.status_path)
    status_jsonl = Path(args.status_jsonl_path)
    result_path = output_dir / "results_vsi_mcq.jsonl"
    summary_path = output_dir / "summary.json"

    pruned_ids = load_pruned_ids(Path(args.pruned_ids))
    source_rows = load_mcq_rows(Path(args.test_jsonl), pruned_ids, args.subset, args.sample_limit)
    rows = shard_rows(source_rows, args.shard_index, args.shard_count)
    existing_results = load_jsonl(result_path) if args.resume and result_path.exists() else []
    completed_ids = {
        str(row.get("id")) for row in existing_results if row.get("id") is not None
    }
    initial_status = {
        "state": "loading_model",
        "output_dir": str(output_dir),
        "result_path": str(result_path),
        "summary_path": str(summary_path),
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path or None,
        "subset": args.subset,
        "source_total_samples": len(source_rows),
        "completed_samples": len(existing_results),
        "total_samples": len(rows),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "accuracy": None,
        "parse_rate": None,
        "debiased_accuracy": None,
    }
    write_status(latest_status, status_jsonl, initial_status)

    if args.resume and len(completed_ids) >= len(rows):
        summary = summarize_results(existing_results)
        summary.update(
            {
                "base_model_path": args.base_model_path,
                "adapter_path": args.adapter_path or None,
                "subset": args.subset,
                "result_path": str(result_path),
                "frame_dir_name": args.frame_dir_name,
                "input_size": args.input_size,
                "tiles_per_frame": 1,
                "mcq_total_source_rows": len(source_rows),
                "mcq_total_shard_rows": len(rows),
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
            }
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_status(
            latest_status,
            status_jsonl,
            {
                **initial_status,
                "state": "completed",
                "completed_samples": len(rows),
                "accuracy": summary["accuracy"],
                "parse_rate": summary["parse_rate"],
                "debiased_accuracy": summary["debiased"]["accuracy"],
            },
        )
        return

    model = AutoModel.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, use_fast=False)
    chat_model = unwrap_chat_model(model)
    device = next(chat_model.parameters()).device

    write_status(latest_status, status_jsonl, {**initial_status, "state": "running", "device": str(device)})

    results: list[dict] = list(existing_results)
    generation_config = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    try:
        result_mode = "a" if args.resume else "w"
        with result_path.open(result_mode, encoding="utf-8") as result_handle:
            for index, row in enumerate(rows, start=1):
                if args.resume and str(row.get("id")) in completed_ids:
                    continue
                frame_paths = resolve_frame_paths(Path(args.frame_cache_root), row, args.frame_dir_name)
                raw_response = ""
                predicted_answer = None
                if frame_paths:
                    pixel_values, num_patches_list = load_frames(frame_paths, args.input_size)
                    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
                    frame_prefix = "".join(
                        f"Frame {frame_index + 1}: <image>\n" for frame_index in range(len(frame_paths))
                    )
                    question = frame_prefix + build_prompt(row, args.answer_instruction)
                    with torch.inference_mode():
                        raw_response = chat_model.chat(
                            tokenizer,
                            pixel_values,
                            question,
                            dict(generation_config),
                            num_patches_list=num_patches_list,
                            history=None,
                            return_history=False,
                        ).strip()
                    predicted_answer = extract_answer_letter(raw_response, row.get("options"))

                gold_answer = str(row.get("ground_truth", "")).strip().upper()
                result_row = {
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "scene_name": row.get("scene_name"),
                    "question_type": row.get("question_type"),
                    "question": row.get("question"),
                    "options": row.get("options"),
                    "correct_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "is_correct": predicted_answer == gold_answer,
                    "raw_response": raw_response,
                    "three_line_valid": bool(THREE_LINE_RE.fullmatch(raw_response)),
                    "frame_paths": frame_paths,
                    "pruned": bool(row.get("pruned")),
                }
                result_handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")
                result_handle.flush()
                results.append(result_row)
                processed_count = len(results)

                if args.empty_cache_every > 0 and processed_count % args.empty_cache_every == 0:
                    torch.cuda.empty_cache()
                if processed_count == 1 or processed_count % args.progress_every == 0 or processed_count == len(rows):
                    metrics = summarize_results(results)
                    write_status(
                        latest_status,
                        status_jsonl,
                        {
                            **initial_status,
                            "state": "running",
                            "completed_samples": processed_count,
                            "accuracy": metrics["accuracy"],
                            "parse_rate": metrics["parse_rate"],
                            "debiased_accuracy": metrics["debiased"]["accuracy"],
                        },
                    )
    except Exception as exc:
        write_status(
            latest_status,
            status_jsonl,
            {**initial_status, "state": "failed", "completed_samples": len(results), "error": repr(exc)},
        )
        raise

    summary = summarize_results(results)
    summary.update(
        {
            "base_model_path": args.base_model_path,
            "adapter_path": args.adapter_path or None,
            "subset": args.subset,
            "result_path": str(result_path),
            "frame_dir_name": args.frame_dir_name,
            "input_size": args.input_size,
            "tiles_per_frame": 1,
            "mcq_total_source_rows": len(source_rows),
            "mcq_total_shard_rows": len(rows),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_status(
        latest_status,
        status_jsonl,
        {
            **initial_status,
            "state": "completed",
            "completed_samples": len(rows),
            "accuracy": summary["accuracy"],
            "parse_rate": summary["parse_rate"],
            "debiased_accuracy": summary["debiased"]["accuracy"],
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"InternVL VSI evaluation failed: {exc!r}", file=sys.stderr)
        raise
