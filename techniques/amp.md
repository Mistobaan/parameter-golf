# Automatic Mixed Precision (AMP)

Automatic Mixed Precision runs numerically sensitive parts of a network in FP32 while letting throughput-heavy ops (matmuls, convolutions, pointwise activations) execute in FP16 or BF16. Framework support such as PyTorch's `torch.autocast` removes most of the manual casting work and provides optional loss scaling for FP16 stability.

## Why use AMP?
- **Higher throughput:** Tensor-core kernels for FP16/BF16 deliver significantly more FLOPs per second than FP32 equivalents.
- **Lower memory use:** Half-precision activations slash memory requirements, enabling larger batches or longer contexts.
- **Minimal code changes:** Wrapping the forward pass with `autocast` and (optionally) a `GradScaler` is usually enough.

## How AMP works
1. **Autocasting:** Within an autocast context, PyTorch dynamically chooses the dtype for each op. Supported ops run in the lower precision; unsupported ones fall back to FP32 automatically.
2. **Loss scaling (for FP16):** A scaling factor multiplies the loss before backpropagation to avoid gradient underflow. After gradients are computed they are unscaled prior to optimizer steps. BF16 often skips this step thanks to its wider exponent range.
3. **Master weights:** Parameters stay in FP32 even if their forward computation used lower precision; gradients are accumulated in FP32 as well.

## Best practices
- **Choose dtype per hardware:** Prefer BF16 on recent GPUs (A100/H100) to avoid loss scaling overhead; stick with FP16 on devices without native BF16.
- **Autocast scope:** Wrap only the forward/micro-batch computation. Optimizer steps, gradient clipping, and logging should occur outside the autocast context.
- **Keep numerically fragile ops in FP32:** Layer-norm statistics, softmax denominators, and loss reductions benefit from higher precision. Autocast handles many of these automatically, but custom kernels should cast to FP32 internally.
- **Watch for overflow/underflow:** If NaNs appear, decrease the global learning rate, enable gradient clipping, or temporarily disable autocast for specific modules.
- **Combine with graph compilation cautiously:** When using `torch.compile`, ensure autocast scopes are preserved (define the compiled module outside the autocast context but call it inside the `with autocast(...)` block).

## Practical PyTorch example
The snippet below demonstrates AMP training for a toy classifier. It automatically falls back to full precision on CPUs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
dtype = torch.bfloat16 if use_amp else torch.float32

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.head = nn.Linear(32, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.head(x)

model = TinyNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

for step in range(200):
    inputs = torch.randn(128, 16, device=device)
    labels = torch.randint(0, 2, (128,), device=device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

model.eval()
with torch.no_grad():
    sample = torch.randn(4, 16, device=device)
    probs = model(sample).softmax(dim=-1)
    print("Predictions:", probs)
```

Running this script on a CUDA device uses BF16 autocast plus mixed-precision gradient scaling; on CPU it silently reverts to full precision. The pattern mirrors how AMP should be integrated into larger training loops.


Choosing the operations to be performed in FP16 precision requires analysis of the numerical behavior of the outputs with respect to inputs of the operation as well as the expected performance benefit. This enables marking operations like matrix multiplies, convolutions and normalization layers as safe, while leaving norm or exp operations as requiring high precision.

**Dynamic loss scaling** enables avoiding both over- and underflows of the gradients during training.

The last few layers of the LLM are more sensitive to the quantization and so we recommend running them in higher precision (for example MXFP8).

[Pretraining Large Transformers in NVFP4](https://arxiv.org/abs/2509.25149v1)

floating-point addition and multiplication are not associative, so even mathematically identical matrix multiplies can give slightly different answers depending on order and implementation.


https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html

When inputs contain large values such that intermediate results may overflow the range of the used datatype, the end result may overflow too, even though it is representable in the original datatype.

A very rough rule is: if your accumulator has unit roundoff u, then a length-N dot product can accumulate error on the order of N u in a worst-case-style view. Higham's matrix multiplication error analysis specifically notes that for very large n, the accumulation term can dominate


earlier layers build better features, later layers combine them linearly.