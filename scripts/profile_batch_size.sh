#!/bin/bash
set -uo pipefail

PYTHON_BIN="$(which python3)"
SWEEP_ID="${SWEEP_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

export PROFILE="${PROFILE:-1}"
export LOADER_PIN_MEMORY="${LOADER_PIN_MEMORY:-1}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export ITERATIONS="${ITERATIONS:-20}"
export MICROBATCH_STEPS="${MICROBATCH_STEPS:-1}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export SEED="${SEED:-1337}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-524288}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600.0}"
export QK_GAIN_INIT="${QK_GAIN_INIT:-1.5}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export NUM_LAYERS="${NUM_LAYERS:-4}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-3}"
export NUM_HEADS="${NUM_HEADS:-3}"
export HEAD_DIM="${HEAD_DIM:-256}"
export MLP_MULT="${MLP_MULT:-3}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
export ROPE_BASE="${ROPE_BASE:-100000.0}"
export LOGIT_SOFTCAP="${LOGIT_SOFTCAP:-10.0}"
export EMBED_LR="${EMBED_LR:-0.6}"
export HEAD_LR="${HEAD_LR:-0.008}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.03}"
export TIED_EMBED_INIT_STD="${TIED_EMBED_INIT_STD:-0.005}"
export MATRIX_LR="${MATRIX_LR:-0.02}"
export SCALAR_LR="${SCALAR_LR:-0.02}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-5}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.85}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-500}"
export BETA1="${BETA1:-0.70}"
export BETA2="${BETA2:-0.95}"
export ADAM_EPS="${ADAM_EPS:-1e-8}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"
export CONTROL_TENSOR_NAME_PATTERNS="${CONTROL_TENSOR_NAME_PATTERNS:-attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights}"
export INT8_KEEP_FLOAT_FP32_NAME_PATTERNS="${INT8_KEEP_FLOAT_FP32_NAME_PATTERNS:-attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export LOCAL_RANK="${LOCAL_RANK:-0}"

export EXPERIMENT_ID="${EXPERIMENT_ID:-profile_batch_size_${SWEEP_ID}}"
BATCH_MULTS="${BATCH_MULTS:-1 2 4 8 16 32 64 128 256}"

status=0
echo "Starting batch-size sweep: sweep_id=${SWEEP_ID} train_seq_len=${TRAIN_SEQ_LEN} mults=${BATCH_MULTS}"

for mult in ${BATCH_MULTS}; do
    export TOKENS_PER_BATCH=$((TRAIN_SEQ_LEN * mult))
    export RUN_ID="${EXPERIMENT_ID}_seq${TRAIN_SEQ_LEN}_mult${mult}_tok${TOKENS_PER_BATCH}_pin${LOADER_PIN_MEMORY}"

    echo
    echo "=== RUN_ID=${RUN_ID} TOKENS_PER_BATCH=${TOKENS_PER_BATCH} (${mult}x TRAIN_SEQ_LEN) ==="
    if ! "${PYTHON_BIN}" scripts/launch_job.py; then
        status=1
        echo "run failed: ${RUN_ID}"
    fi
done

exit "${status}"
