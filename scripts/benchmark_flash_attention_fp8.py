# benchmarks adapted from the flash_attn repository"
""" Useful functions for writing test code. """
import pickle
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

from einops import rearrange, repeat

from flash_attn_interface import flash_attn_func

def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_backward(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Backward pass")
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError("Grad shape does not match output shape")

    def f(*inputs, y, grad):
        # Set .grad to None to avoid extra operation of gradient accumulation
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.grad = None
        y.backward(grad, retain_graph=True)

    t = benchmark.Timer(
        stmt="f(*inputs, y=y, grad=grad)",
        globals={"f": f, "inputs": inputs, "y": y, "grad": grad},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_combined(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward+backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward + Backward pass")
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError("Grad shape does not match output shape")

    def f(grad, *inputs, **kwinputs):
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.grad = None
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            y = fn(*inputs, **kwinputs)
            if type(y) is tuple:
                y = y[0]
        y.backward(grad, retain_graph=True)

    t = benchmark.Timer(
        stmt="f(grad, *inputs, **kwinputs)",
        globals={"f": f, "fn": fn, "inputs": inputs, "grad": grad, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_fwd_bwd(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward+backward pass of an arbitrary function."""
    return (
        benchmark_forward(
            fn,
            *inputs,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_backward(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
    )


def benchmark_all(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward+backward pass of an arbitrary function."""
    return (
        benchmark_forward(
            fn,
            *inputs,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_backward(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_combined(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
    )


def pytorch_profiler(
    fn,
    *inputs,
    trace_filename=None,
    backward=False,
    amp=False,
    amp_dtype=torch.float16,
    cpu=False,
    verbose=True,
    **kwinputs,
):
    """Wrap benchmark functions in Pytorch profiler to see CUDA information."""
    if backward:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
            g = torch.randn_like(out)
    for _ in range(30):  # Warm up
        if backward:
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    x.grad = None
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
        # Backward should be done outside autocast
        if backward:
            out.backward(g, retain_graph=True)
    activities = ([torch.profiler.ProfilerActivity.CPU] if cpu else []) + [
        torch.profiler.ProfilerActivity.CUDA
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        # profile_memory=True,
        with_stack=True,
    ) as prof:
        if backward:
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    x.grad = None
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
        if backward:
            out.backward(g, retain_graph=True)
    if verbose:
        # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
        print(prof.key_averages().table(row_limit=50))
    if trace_filename is not None:
        prof.export_chrome_trace(trace_filename)


def benchmark_memory(fn, *inputs, desc="", verbose=True, **kwinputs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*inputs, **kwinputs)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / ((2**20) * 1000)
    if verbose:
        print(f"{desc} max memory: {mem}GB")
    torch.cuda.empty_cache()
    return mem



def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    elif torch_type == torch.float8_e4m3fn:
        return cudnn.data_type.FP8_E4M3
    elif torch_type == torch.float8_e5m2:
        return cudnn.data_type.FP8_E5M2
    else:
        raise ValueError("Unsupported tensor data type.")

def cudnn_spda_setup(qkv, seqlen_q, seqlen_k, causal=False):
    b, _, _, nheads, headdim = qkv.shape
    assert cudnn is not None, 'CUDNN is not available'
    o_gpu = torch.zeros(b, seqlen_q, nheads, headdim, dtype=qkv.dtype, device=qkv.device)
    o_gpu_transposed = torch.as_strided(
        o_gpu,
        [b, nheads, seqlen_q, headdim],
        [nheads * seqlen_q * headdim, headdim, nheads * headdim, 1],
    )
    stats_gpu = torch.empty(b, nheads, seqlen_q, 1, dtype=torch.float32, device=qkv.device)
    amax_s_gpu = torch.empty(1, 1, 1, 1, dtype=torch.float32, device=qkv.device)
    amax_o_gpu = torch.empty(1, 1, 1, 1, dtype=torch.float32, device=qkv.device)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(qkv.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    new_q = torch.as_strided(
        qkv,
        [b, nheads, seqlen_q, headdim],
        [seqlen_q * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=0,
    )
    q = graph.tensor(
        name = "Q",
        dim = list(new_q.shape),
        stride = list(new_q.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype)
    )
    new_k = torch.as_strided(
        qkv,
        [b, nheads, seqlen_k, headdim],
        [seqlen_k * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=nheads * headdim,
    )
    k = graph.tensor(
        name = "K",
        dim = list(new_k.shape),
        stride = list(new_k.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype)
    )
    new_v = torch.as_strided(
        qkv,
        [b, nheads, seqlen_k, headdim],
        [seqlen_k * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=nheads * headdim * 2,
    )
    v = graph.tensor(
        name = "V",
        dim = list(new_v.shape),
        stride = list(new_v.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype)
    )

    def get_default_scale_tensor():
        return graph.tensor(
            dim = [1, 1, 1, 1],
            stride = [1, 1, 1, 1],
            data_type=cudnn.data_type.FLOAT
        )

    default_scale_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float32, device="cuda")
    descale_q = get_default_scale_tensor()
    descale_k = get_default_scale_tensor()
    descale_v = get_default_scale_tensor()
    descale_s = get_default_scale_tensor()
    scale_s = get_default_scale_tensor()
    scale_o = get_default_scale_tensor()

    o, _, amax_s, amax_o = graph.sdpa_fp8(
        q=q,
        k=k,
        v=v,
        descale_q=descale_q,
        descale_k=descale_k,
        descale_v=descale_v,
        descale_s=descale_s,
        scale_s=scale_s,
        scale_o=scale_o,
        is_inference=True,
        attn_scale=1.0 / math.sqrt(headdim),
        use_causal_mask=causal,
        name="sdpa",
    )

    o.set_output(True).set_dim(o_gpu_transposed.shape).set_stride(o_gpu_transposed.stride())

    amax_s.set_output(False).set_dim(amax_s_gpu.shape).set_stride(amax_s_gpu.stride())
    amax_o.set_output(False).set_dim(amax_o_gpu.shape).set_stride(amax_o_gpu.stride())
    # stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: new_q,
        k: new_k,
        v: new_v,
        descale_q: default_scale_gpu,
        descale_k: default_scale_gpu,
        descale_v: default_scale_gpu,
        descale_s: default_scale_gpu,
        scale_s: default_scale_gpu,
        scale_o: default_scale_gpu,
        o: o_gpu_transposed,
        amax_s: amax_s_gpu,
        amax_o: amax_o_gpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return o_gpu, amax_o_gpu

    return run


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

def time_fwd(func, *args, **kwargs):
    time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean


def main():
    torch.manual_seed(0)

    repeats = 30
    device = 'cuda'
    # dtype = torch.float16
    dtype = torch.float8_e4m3fn

    # bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4224), (2, 8448), (1, 8448 * 2)]
    bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 8192 * 2)]
    # bs_seqlen_vals = [(4, 4096), (2, 8192), (1, 8192 * 2)]
    # bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048)]
    causal_vals = [False, True]
    headdim_vals = [64, 128, 256]
    dim = 2048
    # dim = 256
    dropout_p = 0.0

    methods = (["Pytorch", "Flash3"])

    time_f = {}
    time_b = {}
    time_f_b = {}
    speed_f = {}
    speed_b = {}
    speed_f_b = {}
    
    for causal in causal_vals:
        for headdim in headdim_vals:
            for batch_size, seqlen in bs_seqlen_vals:
                # begin 
                torch.cuda.empty_cache()
                config = (causal, headdim, batch_size, seqlen)
                nheads = dim // headdim
                q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16, requires_grad=False) for _ in range(3)]

                qkv = torch.stack([q, k, v], dim=2) # stack on the nuber of heads: [B, SeqLen, ]
                qkv = qkv.to(torch.bfloat16)
                f = time_fwd(attention_pytorch, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False)
                time_f[config, "Pytorch"] = f
                res_baseline = attention_pytorch(qkv, dropout_p, causal=causal)

                q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
                softmax_scale = q.shape[-1] ** (-0.5)
                descale_q = torch.tensor([1.0], dtype=torch.float32, device='cuda')
                descale_k = torch.tensor([1.0], dtype=torch.float32, device='cuda')
                descale_v = torch.tensor([1.0], dtype=torch.float32, device='cuda')

                f = time_fwd(flash_attn_func, q, k, v, causal=causal, repeats=repeats, verbose=False)
                # f = time_fwd(
                #     _flash_attn_forward,
                #     q, 
                #     k, 
                #     v, 
                #     softmax_scale, 
                #     causal=causal,
                #     window_size=(-1,-1),
                #     descale_q=descale_q, 
                #     descale_k=descale_k, 
                #     descale_v=descale_v, 
                #     repeats=repeats, 
                #     verbose=False
                # )

                # res = flash_attn_func(q, k, v, causal=causal)
                # torch.testing.assert_close(res.half(), res_baseline, atol=0.05, rtol=0.05)

                time_f[config, "Flash3"] = f

                print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
                for method in methods:
                    speed_f[config, method] = efficiency(
                        flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                        time_f[config, method]
                    )
                    #print (time_f[config,method])
                    print(
                        f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, {time_f[config, method] * 1e3} ms, "
                    )


if __name__ == '__main__':
    main()