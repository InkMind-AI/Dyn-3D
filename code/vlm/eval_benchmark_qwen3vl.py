#!/usr/bin/env python3

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ANSWER_INSTRUCTION = "Please answer with the option letter."
ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
ANSWER_WORD_RE = re.compile(r"(?:answer|option)\s*[:：]?\s*([A-D])\b", re.IGNORECASE)
STANDALONE_LETTER_RE = re.compile(r"\b([A-D])\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3-VL model on a sampled benchmark subset.")
    parser.add_argument("--benchmark-path", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-file", default="")
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--status-jsonl-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--answer-instruction", default=ANSWER_INSTRUCTION)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_status(latest_path: Path, jsonl_path: Path, payload: dict) -> None:
    status = dict(payload)
    status.setdefault("timestamp", datetime.now().isoformat())
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = latest_path.with_name(f".{latest_path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(latest_path)
    append_jsonl(jsonl_path, status)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def load_existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            row_id = str(row.get("id", ""))
            if row_id and row_id in seen_ids:
                continue
            if row_id:
                seen_ids.add(row_id)
            rows.append(row)
    return rows


def sample_rows(benchmark_path: Path, sample_size: int, seed: int) -> list[dict]:
    rows = load_jsonl(benchmark_path)
    if sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = rng.sample(range(len(rows)), sample_size)
    return [rows[index] for index in indices]


def resolve_sample_rows(benchmark_path: Path, sample_size: int, seed: int, sample_file_arg: str) -> tuple[list[dict], Path | None]:
    if sample_file_arg:
        sample_path = Path(sample_file_arg)
        if sample_path.exists():
            return load_jsonl(sample_path), sample_path
    return sample_rows(benchmark_path, sample_size, seed), None


def shard_rows(rows: list[dict], shard_index: int, shard_count: int) -> list[dict]:
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")
    if shard_count == 1:
        return rows
    return rows[shard_index::shard_count]


def resolve_frame_paths(frames_root: Path, row: dict) -> list[str]:
    scene_id = str(row.get("scene_id", "")).strip()
    trajectory_name = str(row.get("trajectory_name", "")).strip()
    frame_dir = frames_root / scene_id / f"video_{trajectory_name}"
    frame_paths = sorted(str(path) for path in frame_dir.glob("frame_*.jpg"))
    return frame_paths


def extract_answer_letter(response_text: str) -> str | None:
    if not response_text:
        return None
    match = ANSWER_TAG_RE.search(response_text)
    if match:
        return match.group(1).upper()
    match = ANSWER_WORD_RE.search(response_text)
    if match:
        return match.group(1).upper()
    stripped = response_text.strip()
    if stripped in {"A", "B", "C", "D"}:
        return stripped
    matches = STANDALONE_LETTER_RE.findall(stripped)
    if matches:
        return matches[-1].upper()
    return None


def summarize_results(records: list[dict]) -> dict:
    total = len(records)
    parsed = sum(1 for row in records if row.get("predicted_answer") is not None)
    correct = sum(1 for row in records if row.get("is_correct"))
    by_type = defaultdict(lambda: {"total": 0, "parsed": 0, "correct": 0})

    for row in records:
        bucket = by_type[row["type_name"]]
        bucket["total"] += 1
        bucket["parsed"] += int(row.get("predicted_answer") is not None)
        bucket["correct"] += int(bool(row.get("is_correct")))

    ranked = []
    for type_name, metrics in by_type.items():
        total_count = metrics["total"]
        ranked.append(
            {
                "type_name": type_name,
                "total": total_count,
                "parsed": metrics["parsed"],
                "correct": metrics["correct"],
                "accuracy": metrics["correct"] / total_count if total_count else 0.0,
                "parse_rate": metrics["parsed"] / total_count if total_count else 0.0,
            }
        )
    ranked.sort(key=lambda item: (-item["total"], item["type_name"]))

    return {
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parse_rate": parsed / total if total else 0.0,
        "by_type": ranked,
    }


def build_messages(frame_paths: list[str], question: str, answer_instruction: str) -> list[dict]:
    text = question.strip()
    if answer_instruction.strip():
        text = f"{text}\n\n{answer_instruction.strip()}"
    return [
        {
            "role": "user",
            "content": [{"type": "image", "image": frame_path} for frame_path in frame_paths]
            + [{"type": "text", "text": text}],
        }
    ]


def main() -> None:
    args = parse_args()
    benchmark_path = Path(args.benchmark_path)
    frames_root = Path(args.frames_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_status = Path(args.status_path)
    status_jsonl = Path(args.status_jsonl_path)
    provided_sample_file = Path(args.sample_file) if args.sample_file else None
    sample_output_file = output_dir / "sampled_rows.jsonl"
    result_path = output_dir / "results_sft.jsonl"
    summary_path = output_dir / "summary.json"
    existing_results = load_existing_results(result_path) if args.resume else []
    completed_ids = {str(row.get("id", "")) for row in existing_results if row.get("id") is not None}

    sampled_rows, sample_source_file = resolve_sample_rows(
        benchmark_path,
        args.sample_size,
        args.seed,
        args.sample_file,
    )
    sampled_rows = shard_rows(sampled_rows, args.shard_index, args.shard_count)
    if provided_sample_file is not None and sample_source_file is None:
        sample_output_file = provided_sample_file
    with sample_output_file.open("w", encoding="utf-8") as handle:
        for row in sampled_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    initial_payload = {
        "state": "prepared" if args.prepare_only else "loading_model",
        "output_dir": str(output_dir),
        "sample_file": str(sample_output_file),
        "sample_source_file": str(sample_source_file) if sample_source_file is not None else None,
        "result_path": str(result_path),
        "summary_path": str(summary_path),
        "adapter_path": args.adapter_path or None,
        "base_model_path": args.base_model_path,
        "completed_samples": len(existing_results),
        "total_samples": len(sampled_rows),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "accuracy": None,
        "parse_rate": None,
        "resume": args.resume,
    }
    write_status(latest_status, status_jsonl, initial_payload)

    if args.prepare_only:
        return

    if args.resume and len(completed_ids) >= len(sampled_rows):
        summary = summarize_results(existing_results)
        summary.update(
            {
                "adapter_path": args.adapter_path or None,
                "base_model_path": args.base_model_path,
                "sample_file": str(sample_output_file),
                "sample_source_file": str(sample_source_file) if sample_source_file is not None else None,
                "result_path": str(result_path),
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
            }
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_status(
            latest_status,
            status_jsonl,
            {
                **initial_payload,
                "state": "completed",
                "completed_samples": len(sampled_rows),
                "accuracy": summary["accuracy"],
                "parse_rate": summary["parse_rate"],
            },
        )
        return

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.base_model_path, max_pixels=args.max_pixels)
    device = next(model.parameters()).device

    write_status(
        latest_status,
        status_jsonl,
        {
            **initial_payload,
            "state": "running",
        },
    )

    results: list[dict] = list(existing_results)
    processed_count = len(results)
    result_mode = "a" if args.resume else "w"
    with result_path.open(result_mode, encoding="utf-8") as result_handle:
        for index, row in enumerate(sampled_rows, start=1):
            if args.resume and str(row.get("id", "")) in completed_ids:
                continue
            frame_paths = resolve_frame_paths(frames_root, row)
            raw_response = ""
            predicted_answer = None
            if frame_paths:
                messages = build_messages(frame_paths, row["question"], args.answer_instruction)
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(device)
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                raw_response = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                predicted_answer = extract_answer_letter(raw_response)

            result_row = {
                "id": row["id"],
                "type_name": row["type_name"],
                "question": row["question"],
                "correct_answer": row["correct_answer"],
                "predicted_answer": predicted_answer,
                "is_correct": predicted_answer == row["correct_answer"],
                "raw_response": raw_response,
                "frame_paths": frame_paths,
            }
            result_handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            result_handle.flush()
            results.append(result_row)
            completed_ids.add(str(row["id"]))
            processed_count += 1

            if args.empty_cache_every > 0 and processed_count % args.empty_cache_every == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if processed_count == 1 or processed_count % args.progress_every == 0 or processed_count == len(sampled_rows):
                metrics = summarize_results(results)
                write_status(
                    latest_status,
                    status_jsonl,
                    {
                        "state": "running",
                        "output_dir": str(output_dir),
                        "sample_file": str(sample_output_file),
                        "sample_source_file": str(sample_source_file) if sample_source_file is not None else None,
                        "result_path": str(result_path),
                        "summary_path": str(summary_path),
                        "adapter_path": args.adapter_path or None,
                        "base_model_path": args.base_model_path,
                        "completed_samples": processed_count,
                        "total_samples": len(sampled_rows),
                        "shard_index": args.shard_index,
                        "shard_count": args.shard_count,
                        "accuracy": metrics["accuracy"],
                        "parse_rate": metrics["parse_rate"],
                    },
                )

    summary = summarize_results(results)
    summary.update(
        {
            "adapter_path": args.adapter_path or None,
            "base_model_path": args.base_model_path,
            "sample_file": str(sample_output_file),
            "sample_source_file": str(sample_source_file) if sample_source_file is not None else None,
            "result_path": str(result_path),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_status(
        latest_status,
        status_jsonl,
        {
            "state": "completed",
            "output_dir": str(output_dir),
            "sample_file": str(sample_output_file),
            "sample_source_file": str(sample_source_file) if sample_source_file is not None else None,
            "result_path": str(result_path),
            "summary_path": str(summary_path),
            "adapter_path": args.adapter_path or None,
            "base_model_path": args.base_model_path,
            "completed_samples": len(sampled_rows),
            "total_samples": len(sampled_rows),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "accuracy": summary["accuracy"],
            "parse_rate": summary["parse_rate"],
        },
    )


if __name__ == "__main__":
    main()
