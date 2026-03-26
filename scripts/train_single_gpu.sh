#!/bin/bash
set -euo pipefail

PYTHON_BIN="$(which python3)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# Keep exactly one visible GPU so launch_job.py stays on a single worker even on multi-GPU hosts.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export LOADER_PIN_MEMORY="${LOADER_PIN_MEMORY:-1}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export SEED="${SEED:-1337}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-524288}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-100}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export ITERATIONS="${ITERATIONS:-1000}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export TOKENS_PER_BATCH=$((TRAIN_SEQ_LEN * 296))

# Note: in the current train_gpt.py, MICROBATCH_STEPS affects both grad accumulation
# count and loader batch sizing. Keeping this at 8 matches the requested accumulation
# behavior, but it does not give a perfectly apples-to-apples multi-GPU batch match
# unless TOKENS_PER_BATCH is adjusted accordingly.
export MICROBATCH_STEPS="${MICROBATCH_STEPS:-8}"

export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600.0}"
export QK_GAIN_INIT="${QK_GAIN_INIT:-1.5}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export NUM_LAYERS="${NUM_LAYERS:-4}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-3}"
export NUM_HEADS="${NUM_HEADS:-3}"
export HEAD_DIM="${HEAD_DIM:-256}"
export MLP_MULT="${MLP_MULT:-1}"
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

export EXPERIMENT_ID="${EXPERIMENT_ID:-train_single_gpu}"
export RUN_ID="${RUN_ID:-${EXPERIMENT_ID}_${STAMP}_seq${TRAIN_SEQ_LEN}_tok${TOKENS_PER_BATCH}_acc${MICROBATCH_STEPS}}"
export TRACKIO_PROJECT="train-${NUM_LAYERS}-${MLP_MULT}"

echo "Launching single-GPU training"
echo "RUN_ID=${RUN_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN} TOKENS_PER_BATCH=${TOKENS_PER_BATCH} MICROBATCH_STEPS=${MICROBATCH_STEPS}"

"${PYTHON_BIN}" scripts/launch_job.py
