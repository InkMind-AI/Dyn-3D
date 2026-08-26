#!/usr/bin/env bash
set -euo pipefail

# Reproduce the released InternVL3.5 SFT initialization.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the InternVL3_5-8B-Instruct base model.}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/internvl_sft}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1}"
MASTER_PORT="${MASTER_PORT:-29632}"

python3 "$ROOT/code/vlm/relocate_paths.py" --root "$ROOT" --in-place
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  "$ROOT/code/vlm/scripts/train_internvl_answeronly.py" \
  --model-path "$MODEL_PATH" \
  --dataset "$ROOT/data/sft/sft_train.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --num-train-epochs 1 \
  --learning-rate 2e-5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --warmup-ratio 0.03 \
  --weight-decay 0.05 \
  --max-length 4096 \
  --eval-ratio 0.05 \
  --logging-steps 10 \
  --eval-steps 25 \
  --save-steps 25 \
  --save-total-limit 3 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --seed 42
