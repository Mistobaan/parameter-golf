"""
Parameter-golf training job — runs on Hugging Face infrastructure.
Launched by launch_job.py via run_uv_job().
"""
import datetime, json, os, re, subprocess, sys, tempfile
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
    "CONTROL_TENSOR_NAME_PATTERNS",
    "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
)


def log(msg: str):
    print(f"[{datetime.datetime.now(datetime.UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    trackio_space = os.environ.get("TRACKIO_SPACE_ID")
    experiment = os.environ.get('EXPERIMENT_ID')
    
    ngpus = torch.cuda.device_count()
    
    run = trackio.init(
        project="parameter-golf",
        # name=run_id,
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
    log_path = Path('logs') / f"{run_id}.log"
    log_lines = []
 
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

            # Parse and log training metrics to Trackio in real time
            step_m = re.search(
                r"step:(\d+)/\d+ train_loss:([\d.]+) train_time:(\d+)ms step_avg:(\d+)ms", line
            )
            if step_m:
                step = int(step_m.group(1))
                trackio.log({
                    "train/loss": float(step_m.group(2)),
                    "train/time_ms": int(step_m.group(3)),
                    "train/step_avg_ms": int(step_m.group(4)),
                }, step=step)

            val_m = re.search(
                r"step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+)", line
            )
            if val_m:
                step = int(val_m.group(1))
                trackio.log({
                    "val/loss": float(val_m.group(2)),
                    "val/bpb": float(val_m.group(3)),
                }, step=step)

    proc.wait()
    all_output = "".join(log_lines)
    log(f'parsed {len(log_lines)} lines')
    # Parse final metrics from captured output
    val_bpb = val_loss = bytes_total = training_time_ms = None
    for line in all_output.splitlines():
        m = re.search(r"final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)", line)
        if m:
            val_loss, val_bpb = float(m.group(1)), float(m.group(2))
        m2 = re.search(r"Total submission size int8\+zlib: (\d+) bytes", line)
        if m2:
            bytes_total = int(m2.group(1))
        m3 = re.search(r"train_time:(\d+)ms", line)
        if m3:
            training_time_ms = int(m3.group(1))

    # Log final summary to Trackio
    summary = {"val_bpb": val_bpb, "val_loss": val_loss, "bytes_total": bytes_total,
               "training_time_ms": training_time_ms, "exit_code": proc.returncode}
    trackio.log({f"final/{k}": v for k, v in summary.items() if v is not None})
    trackio.finish()

    results = {
        "val_loss":          val_loss,
        "val_bpb":           val_bpb,
        "bytes_total":       bytes_total,
        "training_time_ms":  training_time_ms,
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
        
    # batch_bucket_files(bucket_id, add=files_to_upload)
    # update_config_status("COMPLETED" if proc.returncode == 0 else "ERROR")
    print(f"Done. val_bpb={val_bpb}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
