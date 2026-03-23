These functions control which implementation of Scaled Dot Product Attention (SDPA) PyTorch uses on CUDA devices. Your configuration enables only Flash Attention while disabling all other backends.

## Function Explanations

| Function | Backend Controlled | Purpose |
|----------|-------------------|---------|
| `enable_cudnn_sdp(False)` | cuDNN Attention | Disables NVIDIA's cuDNN-optimized attention kernel [1](#0-0)  |
| `enable_flash_sdp(True)` | Flash Attention | Enables Flash Attention kernel for faster memory-efficient attention [2](#0-1)  |
| `enable_mem_efficient_sdp(False)` | Memory-Efficient Attention | Disables the memory-efficient attention implementation [3](#0-2)  |
| `enable_math_sdp(False)` | Math Implementation | Disables PyTorch's native C++ math fallback implementation [4](#0-3)  |

## Backend Selection Process

PyTorch selects SDPA backends in priority order: Flash Attention → Memory-Efficient → Math fallback [5](#0-4) . Your configuration forces Flash Attention to be the only available option.

## Implementation Details

These Python functions call into C++ via `torch._C._set_sdp_use_*` methods [6](#0-5) , which set flags in the global context [7](#0-6) . The backend selection logic checks these flags when determining which kernel to use [8](#0-7) .

## Notes

- Flash Attention provides significant speedup and memory savings but has input constraints [9](#0-8) 
- If Flash Attention cannot run due to unsupported inputs, your code will fail since you've disabled all fallback options
- Consider using the context manager `torch.nn.attention.sdpa_kernel` for temporary backend changes instead of global flags [10](#0-9) 

Wiki pages you might want to explore:
- [Overview (pytorch/pytorch)](/wiki/pytorch/pytorch#1)
- [TorchDynamo Frontend (pytorch/pytorch)](/wiki/pytorch/pytorch#2.2)

### Citations

**File:** torch/backends/cuda/__init__.py (L397-403)
```python
def enable_flash_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables flash scaled dot product attention.
    """
    torch._C._set_sdp_use_flash(enabled)
```

**File:** torch/backends/cuda/__init__.py (L415-421)
```python
def enable_mem_efficient_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables memory efficient scaled dot product attention.
    """
    torch._C._set_sdp_use_mem_efficient(enabled)
```

**File:** torch/backends/cuda/__init__.py (L433-439)
```python
def enable_math_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables math scaled dot product attention.
    """
    torch._C._set_sdp_use_math(enabled)
```

**File:** torch/backends/cuda/__init__.py (L542-548)
```python
def enable_cudnn_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables cuDNN scaled dot product attention.
    """
    torch._C._set_sdp_use_cudnn(enabled)
```

**File:** torch/backends/cuda/__init__.py (L551-589)
```python
@contextlib.contextmanager
@deprecated(
    (
        "`torch.backends.cuda.sdp_kernel()` is deprecated. "
        "In the future, this context manager will be removed. "
        "Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, "
        "with updated signature."
    ),
    category=FutureWarning,
)
def sdp_kernel(
    enable_flash: bool = True,
    enable_math: bool = True,
    enable_mem_efficient: bool = True,
    enable_cudnn: bool = True,
):
    r"""
    .. warning:: This flag is beta and subject to change.

    This context manager can be used to temporarily enable or disable any of the three backends for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """
    from torch.nn.attention import sdpa_kernel

    backend_list = []
    if enable_flash:
        backend_list.append(SDPBackend.FLASH_ATTENTION)
    if enable_mem_efficient:
        backend_list.append(SDPBackend.EFFICIENT_ATTENTION)
    if enable_math:
        backend_list.append(SDPBackend.MATH)
    if enable_cudnn:
        backend_list.append(SDPBackend.CUDNN_ATTENTION)

    with sdpa_kernel(backend_list) as context:
        try:
            yield context
        finally:
            pass
```

**File:** aten/src/ATen/native/transformers/cuda/sdp_utils.cpp (L944-981)
```cpp
SDPBackend select_sdp_backend(sdp_params const& kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention
  // 2. Mem Efficient Attention
  // 3. Math fallback
  auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP() &&
      !ctx.userEnabledMemEfficientSDP() && !ctx.userEnabledCuDNNSDP()) {
    return SDPBackend::error;
  }
  // Get ideal kernel ordering
  const auto ordering = priority_order(kernel_params);

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : ordering) {
    switch (backend) {
      case SDPBackend::cudnn_attention:
        if (sdp::can_use_cudnn_attention(kernel_params, print_debug)) {
              return SDPBackend::cudnn_attention;
        }
        break;
      case SDPBackend::flash_attention:
        if (sdp::can_use_flash_attention(kernel_params, print_debug)) {
          return SDPBackend::flash_attention;
        }
        break;
      case SDPBackend::efficient_attention:
        if (sdp::can_use_mem_efficient_attention(kernel_params, print_debug)) {
          return SDPBackend::efficient_attention;
        }
        break;
      case SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return SDPBackend::math;
        }
        break;
```

**File:** torch/csrc/Module.cpp (L921-931)
```cpp
static PyObject* THPModule_setSDPUseFlash(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_sdp_use_math expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setSDPUseFlash(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
```

**File:** aten/src/ATen/Context.cpp (L270-288)
```cpp
void Context::setSDPUseMath(bool e) {
  enabled_mathSDP = e;
}

bool Context::allowFP16BF16ReductionMathSDP() const {
  return allow_fp16_bf16_reduction_mathSDP;
}

void Context::setAllowFP16BF16ReductionMathSDP(bool e) {
  allow_fp16_bf16_reduction_mathSDP = e;
}

bool Context::userEnabledCuDNNSDP() const {
  return enabled_cudnnSDP;
}

void Context::setSDPUseCuDNN(bool e) {
  enabled_cudnnSDP = e;
}
```

**File:** torch/nn/functional.py (L6034-6034)
```python
        There are currently three supported implementations of scaled dot product attention:
```

---

These settings control TensorFloat-32 (TF32) precision usage on NVIDIA Ampere and later GPUs, providing significant performance improvements at the cost of some numerical precision.

## TF32 Settings Explained

| Setting | Controls | Purpose |
|---------|----------|---------|
| `torch.backends.cuda.matmul.allow_tf32 = True` | Matrix Multiplications | Enables TF32 tensor cores for CUDA matrix operations via cuBLAS [1](#1-0)  |
| `torch.backends.cudnn.allow_tf32 = True` | cuDNN Operations | Enables TF32 tensor cores for cuDNN convolutions and other operations [2](#1-1)  |

## Technical Details

TF32 is a precision format that:
- Uses 10 bits of mantissa (vs 23 for FP32)
- Maintains FP32 dynamic range
- Provides ~7x speedup on A100 for matrix operations [3](#1-2) 

## Implementation

The settings are managed through the global context:
- `allowTF32CuBLAS()` checks the matmul setting [4](#1-3) 
- `setAllowTF32CuBLAS()` sets the matmul setting [5](#1-4) 
- cuDNN's `allow_tf32` property uses `ContextProp` for get/set operations [6](#1-5) 

## Usage Context

These flags are commonly used in:
- Benchmarking to control precision vs performance tradeoffs
- Testing to ensure numerical accuracy
- Production environments where performance is critical [7](#1-6) 

## Notes

- PyTorch 1.12+ defaults `allow_tf32` to `False` for matmul but `True` for cuDNN [8](#1-7) 
- New fine-grained precision APIs are being introduced to replace these legacy flags [9](#1-8) 
- Context managers like `tf32_off()` provide temporary control for testing [10](#1-9) 

Wiki pages you might want to explore:
- [Overview (pytorch/pytorch)](/wiki/pytorch/pytorch#1)
- [CUDA Backend (pytorch/pytorch)](/wiki/pytorch/pytorch#3.2)
- [Attention Mechanisms and Transformers (pytorch/pytorch)](/wiki/pytorch/pytorch#3.5)

### Citations

**File:** docs/source/backends.md (L54-58)
```markdown
.. attribute::  allow_tf32

    A :class:`bool` that controls whether TensorFloat-32 tensor cores may be used in matrix
    multiplications on Ampere or newer GPUs. allow_tf32 is going to be deprecated. See :ref:`tf32_on_ampere`.
```
```

**File:** docs/source/backends.md (L197-201)
```markdown
.. attribute::  allow_tf32

    A :class:`bool` that controls where TensorFloat-32 tensor cores may be used in cuDNN
    convolutions on Ampere or newer GPUs. allow_tf32 is going to be deprecated. See :ref:`tf32_on_ampere`.
```
```

**File:** docs/source/notes/cuda.rst (L67-104)
```text
After Pytorch 2.9, we provide a new sets of APIs to control the TF32 behavior in a more fine-grained way, and
suggest to use the new APIs for better control.
We can set float32 precision per backend and per operators. We can also override the global setting for a specific operator.

.. code:: python

  torch.backends.fp32_precision = "ieee"
  torch.backends.cuda.matmul.fp32_precision = "ieee"
  torch.backends.cudnn.fp32_precision = "ieee"
  torch.backends.cudnn.conv.fp32_precision = "tf32"
  torch.backends.cudnn.rnn.fp32_precision = "tf32"

The fp32_precision can be set to `ieee` or `tf32` for `cuda/cudnn`.
`ieee` fp32_precision indicate that we will use `FP32` as internal computation precision.
`tf32` fp32_precision indicate that we will allow to use `TF32` as internal computation precision.

We can override a generic setting for a specific operator if the fp32_precision is set to `ieee`.

.. code:: python

  torch.backends.cudnn.fp32_precision = "tf32"
  torch.backends.cudnn.conv.fp32_precision = "ieee"
  torch.backends.cudnn.rnn.fp32_precision = "ieee"

We can also override a generic setting for a specific backend if the fp32_precision is set to `ieee`.

.. code:: python

  torch.backends.fp32_precision = "tf32"
  torch.backends.cudnn.fp32_precision = "ieee"
  torch.backends.cudnn.conv.fp32_precision = "ieee"
  torch.backends.cudnn.rnn.fp32_precision = "ieee"

For above 2 cases, both `torch.backends.cudnn.conv.fp32_precision` and `torch.backends.cudnn.rnn.fp32_precision`
is overridden to `ieee`.

We suggest to use the new settings for better control. And we do not support to use mix of old and new settings.

```

**File:** docs/source/notes/cuda.rst (L110-112)
```text
Starting in PyTorch 1.7, there is a new flag called `allow_tf32`. This flag
defaults to True in PyTorch 1.7 to PyTorch 1.11, and False in PyTorch 1.12 and later.
This flag controls whether PyTorch is allowed to use the TensorFloat32 (TF32) tensor cores,
```

**File:** docs/source/notes/cuda.rst (L116-164)
```text
TF32 tensor cores are designed to achieve better performance on matmul and convolutions on
`torch.float32` tensors by rounding input data to have 10 bits of mantissa, and accumulating
results with FP32 precision, maintaining FP32 dynamic range.

matmuls and convolutions are controlled separately, and their corresponding flags can be accessed at:

.. code:: python

  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
  # in PyTorch 1.12 and later.
  torch.backends.cuda.matmul.allow_tf32 = True

  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
  torch.backends.cudnn.allow_tf32 = True

The precision of matmuls can also be set more broadly (limited not just to CUDA) via :meth:`~torch.set_float32_matmul_precision`.
Note that besides matmuls and convolutions themselves, functions and nn modules that internally uses
matmuls or convolutions are also affected. These include `nn.Linear`, `nn.Conv*`, cdist, tensordot,
affine grid and grid sample, adaptive log softmax, GRU and LSTM.

To get an idea of the precision and speed, see the example code and benchmark data (on A100) below:

.. code:: python

  a_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
  b_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
  ab_full = a_full @ b_full
  mean = ab_full.abs().mean()  # 80.7277

  a = a_full.float()
  b = b_full.float()

  # Do matmul at TF32 mode.
  torch.backends.cuda.matmul.allow_tf32 = True
  ab_tf32 = a @ b  # takes 0.016s on GA100
  error = (ab_tf32 - ab_full).abs().max()  # 0.1747
  relative_error = error / mean  # 0.0022

  # Do matmul with TF32 disabled.
  torch.backends.cuda.matmul.allow_tf32 = False
  ab_fp32 = a @ b  # takes 0.11s on GA100
  error = (ab_fp32 - ab_full).abs().max()  # 0.0031
  relative_error = error / mean  # 0.000039

From the above example, we can see that with TF32 enabled, the speed is ~7x faster on A100, and that
relative error compared to double precision is approximately 2 orders of magnitude larger. Note that
the exact ratio of TF32 to single precision speed depends on the hardware generation, as properties
such as the ratio of memory bandwidth to compute as well as the ratio of TF32 to FP32 matmul throughput
may vary from generation to generation or model to model.
```

**File:** aten/src/ATen/Context.cpp (L322-332)
```cpp
bool Context::allowTF32CuBLAS() const {
  bool legacy_allow_tf32 = float32_matmul_precision != at::Float32MatmulPrecision::HIGHEST;
  bool allow_tf32_new = float32Precision(Float32Backend::CUDA, Float32Op::MATMUL) == Float32Precision::TF32;
  TORCH_CHECK(
      legacy_allow_tf32 == allow_tf32_new,
      "PyTorch is checking whether allow_tf32_new is enabled for cuBlas matmul,",
      "Current status indicate that you have used mix of the legacy and new APIs to set the TF32 status for cublas matmul. ",
      "We suggest only using the new API to set the TF32 flag. See also: ",
      "https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices");
  return allow_tf32_new;
}
```

**File:** aten/src/ATen/Context.cpp (L334-337)
```cpp
void Context::setAllowTF32CuBLAS(bool b) {
  float32_matmul_precision = b ? at::Float32MatmulPrecision::HIGH : at::Float32MatmulPrecision::HIGHEST;
  setFloat32Precision(Float32Backend::CUDA, Float32Op::MATMUL, b ? Float32Precision::TF32 : Float32Precision::IEEE);
}
```

**File:** torch/backends/cudnn/__init__.py (L230-232)
```python
    allow_tf32 = ContextProp(
        torch._C._get_cudnn_allow_tf32, torch._C._set_cudnn_allow_tf32
    )
```

**File:** benchmarks/dynamo/common.py (L100-102)
```python
# We are primarily interested in TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
```

**File:** torch/testing/_internal/common_cuda.py (L218-227)
```python
@contextlib.contextmanager
def tf32_off():
    old_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32_matmul

```


https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html

