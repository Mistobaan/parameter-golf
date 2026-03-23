# Spectral Weight Compression (PR #503)

## Goal Context
- README objective: maximize compression performance (low val bpb) under 16 MB artifact and 10 min on 8×H100s.
- This PR retools `train_gpt.py` from the simple baseline into a SOTA-ready training/eval recipe tailored to that goal.

## Hyperparameters & Schedule Changes
**Intention**: push more throughput per iteration and expose extra knobs (SWA, QAT, pruning, eval stride) for leaderboard-style tuning.

```python
class Hyperparameters:
    ...
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    ...
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 32))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    ...
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    ...
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    prune_pct = float(os.environ.get("PRUNE_PCT", 0.0))
```
**Analysis**: Longer context (2 k tokens) and larger MLPs improve modeling capacity, but also raise compute per step; without further systems optimizations the baseline 10‑minute limit might be missed. The extra switches are powerful for advanced users yet increase complexity for newcomers, counter to the original goal of keeping this script approachable.

## Expressive Embeddings & Attention Mods
**Intention**: add inductive biases (hashed bigram embeddings, SmearGate, shared value embeddings, optional FlashAttention/XSA) to squeeze quality out of the fixed parameter budget.

```python
class CausalSelfAttention(nn.Module):
    ...
    if _HAS_FA3:
        y = flash_attn_3_func(q, k, v, causal=True).contiguous()
    else:
        y = F.scaled_dot_product_attention(...)

class SmearGate(nn.Module):
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    ...
    def bigram_hash(self, tokens):
        mod = self.bigram_vocab_size - 1
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

class GPT(nn.Module):
    self.bigram = BigramHashEmbedding(...)
    self.smear = SmearGate(model_dim)
    ...
    if self.ve_layer_indices:
        self.ve_shared = ValueEmbedding(...)
        self.ve_layer_scales = nn.ParameterList([...])
    if xsa_last_n > 0:
        for i in range(max(0, num_layers - xsa_last_n), num_layers):
            self.blocks[i].attn.use_xsa = True
```
**Analysis**: These modules can materially lower bpb, especially on short-range patterns, while FlashAttention keeps 2 k-token attention tractable. The tradeoff is additional parameters and tuning burden; artifact size pressure grows and misconfigured VE/XSA could destabilize training. Provided quantization keeps the checkpoint under 16 MB, the gains likely outweigh the costs for leaderboard runs.

## Optimizer, Regularization, & QAT Hooks
**Intention**: strengthen optimization via Muon weight decay, more granular parameter groups, gradient clipping, SWA, and optional quantization-aware training.

```python
class Muon(torch.optim.Optimizer):
    ...
    def step(...):
        ...
        if wd > 0.0:
            p.data.mul_(1.0 - lr * wd)
        p.add_(g, alpha=-lr)

CastedLinear._qat_enabled = args.qat_enabled
base_model = GPT(...).to(device).bfloat16()
for module in base_model.modules():
    if isinstance(module, CastedLinear):
        module.float()
...
if base_model.bigram is not None:
    tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
...
optimizer_muon = Muon(..., weight_decay=args.muon_wd)
...
if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
    ...
```
**Analysis**: Higher Muon momentum plus decay and SWA should improve generalization for the beefier architecture. QAT support prepares the model for mixed-precision export but can destabilize training if enabled without calibration. Memory cost rises due to SWA snapshots, so users must ensure they still fit inside runtime/memory budgets.

## Evaluation & Test-Time Training
**Intention**: replace single-pass eval with sliding-window bpb measurement and optional legal test-time training (score chunk before learning from it) to eke out extra compression.

```python
def eval_val_sliding(..., stride: int, ...):
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    ...
    compiled_logits = torch.compile(base_model.forward_logits, ...)
    ...
    return val_loss, bits_per_token * tokens_per_byte

def eval_val_sliding_ttt(...):
    """Legal score-first TTT"""
    ...
    # Freeze everything, then unfreeze last N blocks + norms + head
    ...
    if ttt_optimizer == "adamw":
        optimizer = torch.optim.AdamW(...)
    else:
        optimizer = torch.optim.SGD(...)
```
**Analysis**: Sliding windows with stride 32 allow reporting longer-context compression aligned with leaderboard practices, and the score-first TTT respects the challenge rules. However, both routines add significant evaluation time; without careful bounds they could exceed the 10‑minute evaluation limit, so documentation/run scripts need to enforce that constraint.

## Compression & Export Pipeline (Mixed Int6 + Zstd)
**Intention**: reduce artifact size by magnitude pruning plus mixed int6 per-row quantization compressed with zstd, then verify via an int6 round-trip before reporting sliding/TTT metrics.

```python
if args.prune_pct > 0:
    for k, v in sd_cpu.items():
        if v.ndim == 2 and v.numel() > 65536:
            thresh = torch.quantile(v.abs().float(), args.prune_pct)
            v[v.abs() < thresh] = 0.0
    ...
quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})
...
quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw) if _COMPRESSOR == "zstd" else zlib.compress(quant_raw, 9)
...
quant_state = torch.load(...)
deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
...
sw_val_loss, sw_val_bpb = eval_val_sliding(...)
```
**Analysis**: Int6 + zstd can markedly shrink the artifact, making room for the added modules while staying below 16 MB. The downside is potential accuracy loss without full QAT and the new dependency on `zstandard` (falls back to zlib but at larger size). Magnitude pruning offers another lever yet can quickly hurt quality if `PRUNE_PCT` is mis-set; careful tuning is needed.

## Overall Takeaways
1. Script now mirrors leaderboard recipes rather than a beginner baseline—powerful but much more complex.
2. Before merging into `main`, decide whether such complexity belongs here or should live under `/records` per repo guidance.
3. Ensure the heavier training/eval flow (longer sequences, SWA, sliding eval, TTT) still fits inside both 10‑minute training and evaluation budgets, and document the extra dependencies (FlashAttention, zstd).
