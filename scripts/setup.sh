#!/bin/bash
set -x

sudo apt install build-essential python3-dev  -y

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy sentencepiece huggingface-hub[hf_transfer] datasets tqdm einops ninja
pip install torch torchvision flash-attn-3 --index-url https://download.pytorch.org/whl/cu130

# check
python -c "import torch; print(torch.utils.collect_env.main())"
