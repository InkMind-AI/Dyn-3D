#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded benchmark or VSI-MCQ evaluation results.")
    parser.add_argument("--task", choices=["benchmark", "vsi_mcq"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def basic_metrics(records: list[dict]) -> dict:
    total = len(records)
    parsed = sum(1 for row in records if row.get("predicted_answer") is not None)
    correct = sum(1 for row in records if row.get("is_correct"))
    return {
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parse_rate": parsed / total if total else 0.0,
    }


def benchmark_summary(records: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_type[str(row.get("type_name", "unknown"))].append(row)

    ranked = []
    for type_name, rows in by_type.items():
        ranked.append({"type_name": type_name, **basic_metrics(rows)})
    ranked.sort(key=lambda item: (-item["total"], item["type_name"]))

    return {**basic_metrics(records), "by_type": ranked}


def vsi_summary(records: list[dict]) -> dict:
    def ranked_breakdown(key_name: str) -> list[dict]:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in records:
            buckets[str(row.get(key_name, "unknown"))].append(row)
        ranked = [{key_name: name, **basic_metrics(rows)} for name, rows in buckets.items()]
        ranked.sort(key=lambda item: (-item["total"], item[key_name]))
        return ranked

    debiased = [row for row in records if not row.get("pruned")]
    pruned = [row for row in records if row.get("pruned")]
    answer_counter = Counter(row.get("correct_answer") for row in records)
    return {
        **basic_metrics(records),
        "debiased": basic_metrics(debiased),
        "pruned": basic_metrics(pruned),
        "by_question_type": ranked_breakdown("question_type"),
        "by_dataset": ranked_breakdown("dataset"),
        "answer_distribution": {letter: answer_counter.get(letter, 0) for letter in "ABCD"},
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    result_name = "results_sft.jsonl" if args.task == "benchmark" else "results_vsi_mcq.jsonl"

    merged_by_id: dict[str, dict] = {}
    shard_summaries = []
    for shard_index in range(args.shard_count):
        shard_dir = output_dir / f"shard_{shard_index}"
        rows = load_jsonl(shard_dir / result_name)
        for row in rows:
            merged_by_id[str(row.get("id"))] = row
        summary_path = shard_dir / "summary.json"
        shard_summaries.append(
            {
                "shard_index": shard_index,
                "result_path": str(shard_dir / result_name),
                "summary_path": str(summary_path),
                "total": len(rows),
                "summary": json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None,
            }
        )

    merged_rows = [merged_by_id[key] for key in sorted(merged_by_id)]
    merged_result_path = output_dir / result_name
    write_jsonl(merged_result_path, merged_rows)

    summary = benchmark_summary(merged_rows) if args.task == "benchmark" else vsi_summary(merged_rows)
    summary.update(
        {
            "base_model_path": args.base_model_path,
            "adapter_path": args.adapter_path,
            "result_path": str(merged_result_path),
            "shard_count": args.shard_count,
            "shards": shard_summaries,
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
