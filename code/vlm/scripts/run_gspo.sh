#!/usr/bin/env bash
set -euo pipefail

# Run either released GSPO configuration after merging the corresponding SFT adapter.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the merged SFT model.}"
CONFIG_NAME="${CONFIG_NAME:?Set CONFIG_NAME to a YAML file under code/EasyR1/examples.}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/gspo}"
RUN_TAG="${RUN_TAG:-gspo_$(date -u +%Y%m%d_%H%M%S)}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29731}"
EASYR1="$ROOT/code/EasyR1"
CONFIG="$EASYR1/examples/$CONFIG_NAME"
RENDERED_CONFIG="$OUTPUT_DIR/$(basename "$CONFIG_NAME")"

[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 1; }
python3 "$ROOT/code/vlm/relocate_paths.py" --root "$ROOT" --in-place
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/training_plots"
sed -e "s|__PROJECT_ROOT__|$ROOT|g" -e "s|__MODEL_PATH__|$MODEL_PATH|g" -e "s|__OUTPUT_DIR__|$OUTPUT_DIR|g" "$CONFIG" > "$RENDERED_CONFIG"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export MASTER_ADDR MASTER_PORT
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$EASYR1:${PYTHONPATH:-}"

cd "$EASYR1"
python3 -m verl.trainer.main \
  config="$RENDERED_CONFIG" \
  data.train_files="$ROOT/data/rl/rl_train.jsonl" \
  data.val_files=null \
  worker.actor.model.model_path="$MODEL_PATH" \
  worker.actor.model.tokenizer_path="$MODEL_PATH" \
  worker.reward.reward_function="${ROOT}/code/EasyR1/examples/reward_function/kinematic_gspo.py:compute_score" \
  trainer.experiment_name="$RUN_TAG" \
  trainer.save_checkpoint_path="$OUTPUT_DIR" \
  trainer.load_checkpoint_path=null \
  trainer.find_last_checkpoint=true \
  trainer.n_gpus_per_node=2
