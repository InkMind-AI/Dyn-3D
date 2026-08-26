#!/usr/bin/env python3

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ANSWER_INSTRUCTION = "Please answer with the option letter."
ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
THREE_LINE_RE = re.compile(
    r"^\s*<reasoning>\s*truth:\s*.+?\s+cot:\s*.+?\s+answer:\s*([A-D])\s*"
    r"</reasoning>\s*<answer>\s*\1\s*</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)
ANSWER_WORD_RE = re.compile(r"(?:answer|option)\s*[:：]?\s*([A-D])\b", re.IGNORECASE)
STANDALONE_LETTER_RE = re.compile(r"\b([A-D])\b")
OPTION_PREFIX_RE = re.compile(r"^\s*([A-D])\s*[\).:：、-]?\s*(.*)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL adapters on the VSI-Bench multiple-choice subset.")
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
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument("--frame-dir-name", default="32f")
    parser.add_argument("--answer-instruction", default=ANSWER_INSTRUCTION)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
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
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                # A hard shutdown can leave only the final appended row incomplete.
                continue
    return rows


def load_pruned_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def is_mcq_row(row: dict) -> bool:
    options = row.get("options")
    return isinstance(options, list) and len(options) >= 2


def load_mcq_rows(test_jsonl: Path, pruned_ids: set[str], subset: str, sample_limit: int) -> list[dict]:
    rows: list[dict] = []
    with test_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not is_mcq_row(row):
                continue
            row["pruned"] = str(row.get("id")) in pruned_ids
            if subset == "debiased" and row["pruned"]:
                continue
            if subset == "pruned" and not row["pruned"]:
                continue
            rows.append(row)
            if sample_limit > 0 and len(rows) >= sample_limit:
                break
    return rows


def shard_rows(rows: list[dict], shard_index: int, shard_count: int) -> list[dict]:
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")
    if shard_count == 1:
        return rows
    return rows[shard_index::shard_count]


def resolve_frame_paths(frame_cache_root: Path, row: dict, frame_dir_name: str) -> list[str]:
    scene_dir = frame_cache_root / str(row.get("dataset", "")) / str(row.get("scene_name", ""))
    preferred_dir = scene_dir / frame_dir_name
    if preferred_dir.is_dir():
        return sorted(str(path) for path in preferred_dir.glob("frame_*.jpg"))
    return []


def normalize_option_text(text: str) -> str:
    text = str(text or "").strip().lower()
    match = OPTION_PREFIX_RE.match(text)
    if match:
        text = match.group(2).strip() or match.group(1).strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9.+\-]+", " ", text)
    return " ".join(text.split())


def option_text_answer(response_text: str, options: list | None) -> str | None:
    if not response_text or not options:
        return None
    normalized_response = normalize_option_text(response_text)
    if not normalized_response:
        return None
    candidates = []
    for index, option in enumerate(options):
        letter = chr(ord("A") + index)
        raw = str(option or "")
        prefix_match = OPTION_PREFIX_RE.match(raw)
        if prefix_match:
            letter = prefix_match.group(1).upper()
        normalized_option = normalize_option_text(raw)
        if normalized_option:
            candidates.append((letter, normalized_option))
    exact = [letter for letter, option_text in candidates if normalized_response == option_text]
    if len(exact) == 1:
        return exact[0]
    contained = [
        letter
        for letter, option_text in candidates
        if normalized_response in option_text or option_text in normalized_response
    ]
    if len(contained) == 1:
        return contained[0]
    return None


def extract_answer_letter(response_text: str, options: list | None = None) -> str | None:
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
    mapped = option_text_answer(response_text, options)
    if mapped:
        return mapped
    return None


def build_prompt(row: dict, answer_instruction: str) -> str:
    lines = [str(row.get("question", "")).strip()]
    for option in row.get("options") or []:
        lines.append(str(option).strip())
    if answer_instruction.strip():
        lines.append("")
        lines.append(answer_instruction.strip())
    return "\n".join(lines).strip()


def build_messages(frame_paths: list[str], prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [{"type": "image", "image": frame_path} for frame_path in frame_paths]
            + [{"type": "text", "text": prompt}],
        }
    ]


def compute_bucket_metrics(records: list[dict]) -> dict:
    total = len(records)
    parsed = sum(1 for row in records if row.get("predicted_answer") is not None)
    correct = sum(1 for row in records if row.get("is_correct"))
    three_line_valid = sum(1 for row in records if row.get("three_line_valid"))
    return {
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parse_rate": parsed / total if total else 0.0,
        "three_line_valid": three_line_valid,
        "three_line_rate": three_line_valid / total if total else 0.0,
    }


def build_ranked_breakdown(records: list[dict], key_name: str) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        buckets[str(row.get(key_name, "unknown"))].append(row)
    ranked = []
    for bucket_name, bucket_rows in buckets.items():
        metrics = compute_bucket_metrics(bucket_rows)
        ranked.append({key_name: bucket_name, **metrics})
    ranked.sort(key=lambda item: (-item["total"], item[key_name]))
    return ranked


def summarize_results(records: list[dict]) -> dict:
    overall = compute_bucket_metrics(records)
    debiased_records = [row for row in records if not row.get("pruned")]
    pruned_records = [row for row in records if row.get("pruned")]
    answer_counter = Counter(row.get("correct_answer") for row in records)
    return {
        **overall,
        "debiased": compute_bucket_metrics(debiased_records),
        "pruned": compute_bucket_metrics(pruned_records),
        "by_question_type": build_ranked_breakdown(records, "question_type"),
        "by_dataset": build_ranked_breakdown(records, "dataset"),
        "answer_distribution": {letter: answer_counter.get(letter, 0) for letter in "ABCD"},
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_status = Path(args.status_path)
    status_jsonl = Path(args.status_jsonl_path)
    result_path = output_dir / "results_vsi_mcq.jsonl"
    summary_path = output_dir / "summary.json"

    pruned_ids = load_pruned_ids(Path(args.pruned_ids))
    source_rows = load_mcq_rows(
        test_jsonl=Path(args.test_jsonl),
        pruned_ids=pruned_ids,
        subset=args.subset,
        sample_limit=args.sample_limit,
    )
    rows = shard_rows(source_rows, args.shard_index, args.shard_count)
    existing_results = load_jsonl(result_path) if args.resume and result_path.exists() else []
    completed_ids = {
        str(row.get("id")) for row in existing_results if row.get("id") is not None
    }

    initial_payload = {
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
    write_status(latest_status, status_jsonl, initial_payload)

    if args.resume and len(completed_ids) >= len(rows):
        summary = summarize_results(existing_results)
        summary.update(
            {
                "base_model_path": args.base_model_path,
                "adapter_path": args.adapter_path or None,
                "subset": args.subset,
                "result_path": str(result_path),
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
                **initial_payload,
                "state": "completed",
                "completed_samples": len(rows),
                "accuracy": summary["accuracy"],
                "parse_rate": summary["parse_rate"],
                "debiased_accuracy": summary["debiased"]["accuracy"],
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
            "device": str(device),
        },
    )

    results: list[dict] = list(existing_results)
    result_mode = "a" if args.resume else "w"
    with result_path.open(result_mode, encoding="utf-8") as result_handle:
        for index, row in enumerate(rows, start=1):
            if args.resume and str(row.get("id")) in completed_ids:
                continue
            frame_paths = resolve_frame_paths(Path(args.frame_cache_root), row, args.frame_dir_name)
            raw_response = ""
            predicted_answer = None
            if frame_paths:
                prompt = build_prompt(row, args.answer_instruction)
                messages = build_messages(frame_paths, prompt)
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

            if args.empty_cache_every > 0 and processed_count % args.empty_cache_every == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if processed_count == 1 or processed_count % args.progress_every == 0 or processed_count == len(rows):
                metrics = summarize_results(results)
                write_status(
                    latest_status,
                    status_jsonl,
                    {
                        "state": "running",
                        "output_dir": str(output_dir),
                        "result_path": str(result_path),
                        "summary_path": str(summary_path),
                        "base_model_path": args.base_model_path,
                        "adapter_path": args.adapter_path or None,
                        "subset": args.subset,
                        "source_total_samples": len(source_rows),
                        "completed_samples": processed_count,
                        "total_samples": len(rows),
                        "shard_index": args.shard_index,
                        "shard_count": args.shard_count,
                        "accuracy": metrics["accuracy"],
                        "parse_rate": metrics["parse_rate"],
                        "debiased_accuracy": metrics["debiased"]["accuracy"],
                    },
                )

    summary = summarize_results(results)
    summary.update(
        {
            "base_model_path": args.base_model_path,
            "adapter_path": args.adapter_path or None,
            "subset": args.subset,
            "result_path": str(result_path),
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
            "state": "completed",
            "output_dir": str(output_dir),
            "result_path": str(result_path),
            "summary_path": str(summary_path),
            "base_model_path": args.base_model_path,
            "adapter_path": args.adapter_path or None,
            "subset": args.subset,
            "source_total_samples": len(source_rows),
            "completed_samples": len(rows),
            "total_samples": len(rows),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "accuracy": summary["accuracy"],
            "parse_rate": summary["parse_rate"],
            "debiased_accuracy": summary["debiased"]["accuracy"],
        },
    )


if __name__ == "__main__":
    main()
