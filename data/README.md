---
license: apache-2.0
language:
- en
pretty_name: FineWeb 10B Bytes
task_categories:
- text-generation
---

# FineWeb 10B Bytes

This repository contains training shards for byte-level language model pretraining.

The dataset format is the same format used by [openai/parameter-golf](https://github.com/openai/parameter-golf), the OpenAI Model Craft Challenge repository for training compact language models and evaluating them on FineWeb in bits per byte. In that repository, evaluation is described as tokenizer-agnostic and based on compression performance on the FineWeb validation set.  [oai_citation:0‡GitHub](https://github.com/openai/parameter-golf)

## Origin

These byte shards were generated using the data conversion approach from [openai/parameter-golf Pull Request #705](https://github.com/openai/parameter-golf/pull/705), authored by GitHub user [`seanward`](https://github.com/seanward). That PR is titled **“Byte-Level Tokenizer-Free Transformer”** and explicitly includes a conversion script named `convert_to_bytes.py`, described there as **“Data conversion (sp1024 → raw bytes)”**.  [oai_citation:1‡GitHub](https://github.com/openai/parameter-golf/pull/705)

## Contents

This repository stores shard files such as:

- `fineweb_train_000000.bin`
- `fineweb_train_000001.bin`
- `fineweb_train_000002.bin`

and so on.

## Dataset format

The `.bin` shards follow the same binary training-data convention used for byte-level experiments in `parameter-golf`.

At a high level:

- data is represented as raw UTF-8 bytes
- the byte vocabulary size is 256
- shards are intended for training tokenizer-free / byte-level models
- the data layout is meant for efficient streaming during pretraining

The associated PR #705 describes the model as operating directly on raw UTF-8 bytes with `vocab=256`, and states that it uses raw byte input without BPE or SentencePiece.  [oai_citation:2‡GitHub](https://github.com/openai/parameter-golf/pull/705)

## Provenance

Source data is derived from FineWeb preprocessing workflows associated with byte-level training experiments for `parameter-golf`.

This repository republishes the resulting training shards only. It does **not** bundle the training code itself; for the original training setup, conversion logic, and experiment context, see:

- `openai/parameter-golf`
- PR #705 by `seanward` (“Byte-Level Tokenizer-Free Transformer”)  [oai_citation:3‡GitHub](https://github.com/openai/parameter-golf)

## Intended use

This dataset is intended for:

- byte-level language model pretraining
- tokenizer-free training experiments
- reproducing or adapting `parameter-golf`-style data pipelines
- benchmarking compact models on byte-level objectives
