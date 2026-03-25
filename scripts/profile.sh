#!/bin/bash
set -e

export EXPERIMENT_ID=base_gpt_opt
# train_gpt.py generates a UUID when RUN_ID is unset; mirror that behavior here.
# when using trackio let's use trackio autogen run id
# export RUN_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
# export PROFILE=1
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export SEED=1337
export VAL_BATCH_SIZE=524288
export VAL_LOSS_EVERY=1000
export TRAIN_LOG_EVERY=100
export ITERATIONS=500
export WARMDOWN_ITERS=3500
export WARMUP_STEPS=20
export TRAIN_SEQ_LEN=2048
export TOKENS_PER_BATCH=8192
export MICROBATCH_STEPS=8
export MAX_WALLCLOCK_SECONDS=600.0
export QK_GAIN_INIT=1.5
export VOCAB_SIZE=1024
export NUM_LAYERS=8
export NUM_KV_HEADS=6
export NUM_HEADS=6
export HEAD_DIM=128
export MLP_MULT=3
export TIE_EMBEDDINGS=1
export ROPE_BASE=100000.0
export LOGIT_SOFTCAP=10.0
export EMBED_LR=0.6
export HEAD_LR=0.008
export TIED_EMBED_LR=0.03
export TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.02
export SCALAR_LR=0.02
export MUON_MOMENTUM=0.95
export MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=500
export BETA1=0.70
export BETA2=0.95
export ADAM_EPS=1e-8
export GRAD_CLIP_NORM=0.0
export CONTROL_TENSOR_NAME_PATTERNS=attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights
export INT8_KEEP_FLOAT_FP32_NAME_PATTERNS=attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

# export TRACKIO_SPACE_ID=mistobaan/parameter-golf

$(which python3) scripts/launch_job.py
