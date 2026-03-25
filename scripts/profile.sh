#!/bin/bash
set -e

export EXPERIMENT_ID=base_gpt_opt
export RUN_ID=001 
export PROFILE=1
export ITERATIONS=20
export WARMUP_STEPS=20
export VAL_LOSS_EVERY=0
export DATA_PATH=./data/datasets/fineweb10B_sp1024/ 
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export TOKENS_PER_BATCH=131072
export MINIBATCH_STEPS=1
export NUM_LAYERS=2

# export TRACKIO_SPACE_ID=mistobaan/parameter-golf

$(which python3) scripts/launch_job.py
