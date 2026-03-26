"""
Parameter-golf training job — runs on Hugging Face infrastructure.
Launched by launch_job.py via run_uv_job().
"""
import datetime, json, os, re, subprocess, sys
from pathlib import Path
import trackio
import torch


TRAIN_ENV_KEYS = (
    "DATA_PATH",
    "TOKENIZER_PATH",
    "SEED",
    "VAL_BATCH_SIZE",
    "VAL_LOSS_EVERY",
    "TRAIN_LOG_EVERY",
    "ITERATIONS",
    "WARMDOWN_ITERS",
    "PROFILE",
    "WARMUP_STEPS",
    "TRAIN_SEQ_LEN",
    "TOKENS_PER_BATCH",
    "MICROBATCH_STEPS",
    "MAX_WALLCLOCK_SECONDS",
    "QK_GAIN_INIT",
    "VOCAB_SIZE",
    "NUM_LAYERS",
    "NUM_KV_HEADS",
    "NUM_HEADS",
    "HEAD_DIM",
    "MODEL_DIM",
    "MLP_MULT",
    "TIE_EMBEDDINGS",
    "ROPE_BASE",
    "LOGIT_SOFTCAP",
    "EMBED_LR",
    "HEAD_LR",
    "TIED_EMBED_LR",
    "TIED_EMBED_INIT_STD",
    "MATRIX_LR",
    "SCALAR_LR",
    "MUON_MOMENTUM",
    "MUON_BACKEND_STEPS",
    "MUON_MOMENTUM_WARMUP_START",
    "MUON_MOMENTUM_WARMUP_STEPS",
    "BETA1",
    "BETA2",
    "ADAM_EPS",
    "GRAD_CLIP_NORM",
    "LOADER_PIN_MEMORY",
    "CONTROL_TENSOR_NAME_PATTERNS",
    "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
)

KEYVAL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):(\S+)")
STEP_RE = re.compile(r"\bstep:(\d+)/(\d+)")
INT_RE = re.compile(r"[+-]?\d+")
FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?")
PROFILE_STEP_KEYS = {
    "tok_s",
    "avg_tok_s",
    "gpu_tok_s",
    "avg_gpu_tok_s",
    "loader_ms",
    "loader_pct",
    "host_batch_ms",
    "h2d_submit_ms",
    "shard_io_ms",
    "shard_loads",
    "shard_mb",
    "shard_mb_s",
}


def log(msg: str):
    print(f"[{datetime.datetime.now(datetime.UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def coerce_scalar(raw: str):
    suffix = None
    value_text = raw
    if value_text.endswith("ms"):
        suffix = "ms"
        value_text = value_text[:-2]
    elif value_text.endswith("MiB"):
        suffix = "mib"
        value_text = value_text[:-3]
    elif value_text.endswith("%"):
        suffix = "pct"
        value_text = value_text[:-1]

    lowered = value_text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true", suffix
    if INT_RE.fullmatch(value_text):
        return int(value_text), suffix
    if FLOAT_RE.fullmatch(value_text):
        return float(value_text), suffix
    return None, None


def normalize_metric_name(key: str, suffix: str | None) -> str:
    if suffix == "ms" and not key.endswith("_ms"):
        return f"{key}_ms"
    if suffix == "mib" and not key.endswith("_mib"):
        return f"{key}_mib"
    if suffix == "pct" and not key.endswith("_pct"):
        return f"{key}_pct"
    return key


def parse_step(line: str) -> tuple[int | None, int | None]:
    match = STEP_RE.search(line)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def parse_keyvals(line: str) -> dict[str, int | float | bool]:
    metrics: dict[str, int | float | bool] = {}
    for key, raw in KEYVAL_RE.findall(line):
        if key == "step":
            continue
        value, suffix = coerce_scalar(raw)
        if value is None:
            continue
        metrics[normalize_metric_name(key, suffix)] = value
    return metrics


def remap_metrics(
    metrics: dict[str, int | float | bool],
    aliases: dict[str, str],
    prefix: str,
) -> dict[str, int | float | bool]:
    return {f"{prefix}/{aliases.get(key, key)}": value for key, value in metrics.items()}


def main():
    trackio_project = os.environ.get("TRACKIO_PROJECT", "project")
    trackio_space = os.environ.get("TRACKIO_SPACE_ID")
    experiment = os.environ.get('EXPERIMENT_ID')
    run_id = os.environ.get("RUN_ID", None)
    ngpus = torch.cuda.device_count()

    run = trackio.init(
        project=trackio_project,
        name=run_id,
        auto_log_gpu=ngpus > 0,
        config={
            key.lower(): os.environ.get(key)
            for key in TRAIN_ENV_KEYS
        },
        **({"space_id": trackio_space} if trackio_space else {}),
    )

    run_id = run.name

    # Run training — env vars (NUM_LAYERS, MODEL_DIM, etc.) picked up by Hyperparameters class
    log(f"Starting torchrun (nproc_per_node={ngpus})...")
    os.makedirs("logs", exist_ok=True)
    log_path = Path('logs') / f"{run_id}.log"
    log_lines = []
    tokens_per_batch = env_int("TOKENS_PER_BATCH")
    summary: dict[str, int | float | bool | None] = {
        "val_bpb": None,
        "val_loss": None,
        "bytes_total": None,
        "training_time_ms": None,
        "profile_avg_tok_s": None,
        "profile_avg_gpu_tok_s": None,
        "profile_best_tok_s": None,
        "profile_avg_loader_ms": None,
        "profile_avg_host_batch_ms": None,
        "profile_avg_h2d_submit_ms": None,
        "profile_total_shard_io_ms": None,
        "profile_shard_loads": None,
        "peak_memory_allocated_mib": None,
        "peak_memory_reserved_mib": None,
        "oom": None,
    }

    torchrun_cmd = ["torchrun", f"--nproc_per_node={ngpus}", "train_gpt.py"]

    proc = subprocess.Popen(
        torchrun_cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
        env={ **os.environ.copy(), "RUN_ID": run_id}
    )

    with open(log_path, "w") as log_file:
        stream = proc.stdout if proc.stdout else []
        for line in stream:
            print(line, end="", flush=True)  # stream to job logs live
            log_file.write(line)
            log_lines.append(line)

            step, _total_steps = parse_step(line)
            metrics = parse_keyvals(line)

            if step is not None and "val_loss" in metrics:
                payload = {}
                if "val_loss" in metrics:
                    payload["val/loss"] = float(metrics["val_loss"])
                    summary["val_loss"] = float(metrics["val_loss"])
                if "val_bpb" in metrics:
                    payload["val/bpb"] = float(metrics["val_bpb"])
                    summary["val_bpb"] = float(metrics["val_bpb"])
                if "train_time_ms" in metrics:
                    summary["training_time_ms"] = int(metrics["train_time_ms"])
                if payload:
                    trackio.log(payload, step=step)
                continue

            if step is not None and any(key in metrics for key in PROFILE_STEP_KEYS):
                payload = remap_metrics(
                    metrics,
                    aliases={
                        "train_loss": "loss",
                        "train_time_ms": "time_ms",
                    },
                    prefix="profile",
                )
                if "train_loss" in metrics:
                    payload["train/loss"] = float(metrics["train_loss"])
                if payload:
                    trackio.log(payload, step=step)
                if "train_time_ms" in metrics:
                    summary["training_time_ms"] = int(metrics["train_time_ms"])
                if "avg_tok_s" in metrics:
                    summary["profile_avg_tok_s"] = float(metrics["avg_tok_s"])
                if "avg_gpu_tok_s" in metrics:
                    summary["profile_avg_gpu_tok_s"] = float(metrics["avg_gpu_tok_s"])
                if "tok_s" in metrics:
                    tok_s = float(metrics["tok_s"])
                    current_best = summary["profile_best_tok_s"]
                    summary["profile_best_tok_s"] = tok_s if current_best is None else max(float(current_best), tok_s)
                continue

            if step is not None and "train_loss" in metrics:
                payload = {}
                if "train_loss" in metrics:
                    payload["train/loss"] = float(metrics["train_loss"])
                if "train_time_ms" in metrics:
                    payload["train/time_ms"] = int(metrics["train_time_ms"])
                    summary["training_time_ms"] = int(metrics["train_time_ms"])
                if "step_avg_ms" in metrics:
                    payload["train/step_avg_ms"] = float(metrics["step_avg_ms"])
                if payload:
                    trackio.log(payload, step=step)
                continue

            if line.startswith("profile_summary:"):
                payload = remap_metrics(metrics, aliases={}, prefix="profile")
                if payload:
                    trackio.log(payload)
                if "avg_tok_s" in metrics:
                    summary["profile_avg_tok_s"] = float(metrics["avg_tok_s"])
                if "avg_gpu_tok_s" in metrics:
                    summary["profile_avg_gpu_tok_s"] = float(metrics["avg_gpu_tok_s"])
                if "avg_loader_ms" in metrics:
                    summary["profile_avg_loader_ms"] = float(metrics["avg_loader_ms"])
                if "avg_host_batch_ms" in metrics:
                    summary["profile_avg_host_batch_ms"] = float(metrics["avg_host_batch_ms"])
                if "avg_h2d_submit_ms" in metrics:
                    summary["profile_avg_h2d_submit_ms"] = float(metrics["avg_h2d_submit_ms"])
                if "total_shard_io_ms" in metrics:
                    summary["profile_total_shard_io_ms"] = float(metrics["total_shard_io_ms"])
                if "shard_loads" in metrics:
                    summary["profile_shard_loads"] = int(metrics["shard_loads"])
                if tokens_per_batch > 0 and "avg_tok_s" in metrics:
                    trackio.log({"sweep/avg_tok_s_by_tokens_per_batch": float(metrics["avg_tok_s"])}, step=tokens_per_batch)
                if tokens_per_batch > 0 and "avg_gpu_tok_s" in metrics:
                    trackio.log(
                        {"sweep/avg_gpu_tok_s_by_tokens_per_batch": float(metrics["avg_gpu_tok_s"])},
                        step=tokens_per_batch,
                    )
                continue

            peak_m = re.search(r"peak memory allocated: (\d+) MiB reserved: (\d+) MiB", line)
            if peak_m:
                allocated = int(peak_m.group(1))
                reserved = int(peak_m.group(2))
                summary["peak_memory_allocated_mib"] = allocated
                summary["peak_memory_reserved_mib"] = reserved
                trackio.log(
                    {
                        "system/peak_memory_allocated_mib": allocated,
                        "system/peak_memory_reserved_mib": reserved,
                    }
                )
                continue

            if line.strip() == "oom:true":
                summary["oom"] = True
                trackio.log({"run/oom": 1})
                continue

            final_eval_m = re.search(r"final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)", line)
            if final_eval_m:
                summary["val_loss"] = float(final_eval_m.group(1))
                summary["val_bpb"] = float(final_eval_m.group(2))
                continue

            size_m = re.search(r"Total submission size int8\+zlib: (\d+) bytes", line)
            if size_m:
                summary["bytes_total"] = int(size_m.group(1))
                continue

    proc.wait()
    log(f'parsed {len(log_lines)} lines')
    summary["exit_code"] = proc.returncode
    if tokens_per_batch > 0 and summary["profile_avg_tok_s"] is not None:
        trackio.log({"sweep/final_avg_tok_s_by_tokens_per_batch": float(summary["profile_avg_tok_s"])}, step=tokens_per_batch)

    # Log final summary to Trackio
    trackio.log({f"final/{k}": v for k, v in summary.items() if v is not None})

    results = {
        "val_loss":          summary["val_loss"],
        "val_bpb":           summary["val_bpb"],
        "bytes_total":       summary["bytes_total"],
        "training_time_ms":  summary["training_time_ms"],
        "profile_avg_tok_s": summary["profile_avg_tok_s"],
        "profile_avg_gpu_tok_s": summary["profile_avg_gpu_tok_s"],
        "profile_best_tok_s": summary["profile_best_tok_s"],
        "completed_at":      datetime.datetime.now(datetime.UTC).isoformat() + "Z",
        "exit_code":         proc.returncode,
    }

    model_path = Path("final_model.int8.ptz")
    files_to_upload = [
        (str(log_path),                           f"{experiment}/train.log"),
        (json.dumps(results, indent=2).encode(),  f"{experiment}/results.json"),
    ]
    if model_path.exists():
        files_to_upload.append((str(model_path), f"{experiment}/final_model.int8.ptz"))
        trackio.save(model_path)

    trackio.finish()

    # batch_bucket_files(bucket_id, add=files_to_upload)
    # update_config_status("COMPLETED" if proc.returncode == 0 else "ERROR")
    print(f"Done. val_bpb={summary['val_bpb']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
