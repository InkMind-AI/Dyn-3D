#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def build_answer(row):
    reward_meta = row.get("reward_metadata") or {}
    motion_truth = row.get("motion_truth") or reward_meta.get("motion_truth") or {}
    return {
        "ground_truth_answer": row.get("correct_answer") or reward_meta.get("ground_truth_answer"),
        "question_type": row.get("task_type") or reward_meta.get("question_type"),
        "task_family": row.get("task_family"),
        "answer_text": row.get("answer_text"),
        "options": row.get("options"),
        "gt_value": row.get("gt_value", reward_meta.get("gt_value")),
        "gt_secondary": row.get("gt_secondary", reward_meta.get("gt_secondary")),
        "motion_truth": motion_truth,
        "reward_metadata": reward_meta,
        "sample_id": row.get("sample_id"),
    }


def convert(input_path, train_path, val_path, val_size):
    input_path = Path(input_path)
    train_path = Path(train_path)
    val_path = Path(val_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    val_written = 0
    missing = 0
    with input_path.open("r", encoding="utf-8") as src, \
            train_path.open("w", encoding="utf-8") as train_f, \
            val_path.open("w", encoding="utf-8") as val_f:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            images = row.get("images") or row.get("frame_paths") or []
            prompt = row.get("prompt") or ""
            answer = build_answer(row)
            if not prompt or not images or not answer.get("ground_truth_answer"):
                missing += 1
                continue
            out = {
                "prompt": prompt,
                "images": images,
                "answer": json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
                "sample_id": row.get("sample_id") or row.get("source_row_id"),
                "task_type": row.get("task_type"),
            }
            encoded = json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n"
            train_f.write(encoded)
            if val_written < val_size:
                val_f.write(encoded)
                val_written += 1
            total += 1

    print(json.dumps({
        "input": str(input_path),
        "train": str(train_path),
        "val": str(val_path),
        "train_rows": total,
        "val_rows": val_written,
        "skipped_missing": missing,
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-size", type=int, default=256)
    args = parser.parse_args()
    convert(args.input, args.train_output, args.val_output, args.val_size)


if __name__ == "__main__":
    main()
