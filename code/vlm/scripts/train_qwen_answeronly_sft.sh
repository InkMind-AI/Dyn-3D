#!/usr/bin/env bash
set -euo pipefail

# Reproduce the released Qwen SFT initialization.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the Qwen3-VL-8B-Instruct base model.}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/qwen_sft}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1}"
MASTER_PORT="${MASTER_PORT:-29631}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SWIFT_SFT_PY="${SWIFT_SFT_PY:-}"

python3 "$ROOT/code/vlm/relocate_paths.py" --root "$ROOT" --in-place
mkdir -p "$OUTPUT_DIR"

if [[ -z "$SWIFT_SFT_PY" ]]; then
  SWIFT_SFT_PY="$($PYTHON_BIN -c 'import swift.cli.sft; print(swift.cli.sft.__file__)')"
fi

CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port="$MASTER_PORT" \
  "$SWIFT_SFT_PY" \
  --model "$MODEL_PATH" \
  --model_type qwen3_vl \
  --template qwen3_vl \
  --dataset "$ROOT/data/sft/sft_train.jsonl" \
  --split_dataset_ratio 0.05 \
  --dataset_num_proc 4 \
  --dataset_shuffle true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.05 \
  --lr_scheduler_type cosine \
  --bf16 true \
  --gradient_checkpointing true \
  --eval_steps 25 \
  --save_steps 25 \
  --logging_steps 10 \
  --save_total_limit 2 \
  --report_to none \
  --dataloader_num_workers 4 \
  --max_length 4096 \
  --max_pixels 200704 \
  --freeze_vit true \
  --freeze_aligner true \
  --target_modules all-linear \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --max_new_tokens 32 \
  --output_dir "$OUTPUT_DIR"
