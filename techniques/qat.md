# Quantization Aware Training (QAT)

Quantization Aware Training simulates low-bit arithmetic during optimization so a network learns to tolerate clipping and rounding before it is actually converted to a quantized representation. Fake quantization operators are inserted into the forward pass, while gradients still flow through full-precision master weights.

## Why use QAT?
- **Accuracy retention:** QAT significantly reduces the quality drop typically seen when exporting to int8/int6/int4 compared to post-training quantization.
- **Deployment alignment:** Training with the same constraints the inference hardware imposes avoids unpleasant surprises late in the pipeline.
- **Stable serialization:** Because weights already see quantization noise, exporting to compact formats matches the behavior observed during training.

## How it works
1. **Fake quantization:** During the forward pass, tensors are clamped to a target range and rounded to the nearest representable value, then de-quantized back to floating point for the remainder of the computation.
2. **Straight-through gradients:** Backpropagation bypasses the non-differentiable rounding function by forwarding the incoming gradient (or masking values outside the clip range).
3. **Scale tracking:** Each tensor maintains a scale (per-tensor or per-channel). Scales can be learned or recomputed from statistics such as running maxima.
4. **Optimizer updates:** Optimizers operate on FP32 master weights. Before each forward pass those weights go through fake quantization; after the backward pass the updated master copy is preserved.
5. **Export:** Once converged, fake-quantized weights are replaced with actual low-precision tensors plus their scales.

## Implementation tips
- **Match deployment:** Only quantize the tensors that will be stored or executed in low precision. Extra fake quant nodes just add noise.
- **Prefer per-channel scales:** Per-row or per-column scaling typically yields better accuracy than a single tensor-wide scale, especially for weight matrices.
- **Introduce gradually:** Enable QAT after a short full-precision warmup or ramp the clipping strength to avoid destabilizing early training.
- **Verify with round-trips:** Periodically quantize/dequantize the checkpoint and run evaluation to confirm parity with the fake-quant graph.

## Practical PyTorch example
The following script trains a tiny MLP on synthetic data while simulating int6 quantization for both weights and activations.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

BITS = 6
QMAX = 2 ** (BITS - 1) - 1

@torch.jit.script
def fake_quantize(x: torch.Tensor, per_channel: bool = False) -> torch.Tensor:
    if per_channel and x.dim() >= 2:
        scale = x.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / QMAX
    else:
        scale = x.detach().abs().amax().clamp(min=1e-6) / QMAX
    q = torch.clamp(torch.round(x / scale), -QMAX, QMAX)
    return q * scale

class QLinear(nn.Linear):
    def forward(self, x):
        w_q = fake_quantize(self.weight, per_channel=True)
        x_q = fake_quantize(x)
        out = F.linear(x_q, w_q, self.bias)
        return fake_quantize(out)

class TinyQATModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = QLinear(16, 32)
        self.fc2 = QLinear(32, 16)
        self.head = QLinear(16, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.head(x)

torch.manual_seed(0)
model = TinyQATModel()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(500):
    x = torch.randn(64, 16)
    target = (x.sum(dim=1, keepdim=True) > 0).float()
    pred = model(x)
    loss = F.mse_loss(pred, target)
    opt.zero_grad()
    loss.backward()
    opt.step()

# Export: grab fake-quantized weights for deployment
quantized_state = {k: fake_quantize(v.detach(), per_channel=v.ndim == 2) for k, v in model.state_dict().items()}
print("Sample dequantized weight", quantized_state['fc1.weight'][0, :5])
```

Running the example shows how fake quantization is embedded into the forward pass, how gradients still flow through the FP32 master weights, and how to obtain quantized tensors for export.

## Notes

| Empirically, we found that disabling fake quantization for the first N steps led to better results, presumably because doing so allows the weights to stabilize before we start introducing quantization noise to the fine-tuning process. We disable fake quantization for the first 1000 steps for all our experiments.




## References

- [PyTorch QAT](https://pytorch.org/blog/quantization-aware-training/)
- [TorchAO QAT Documentation](https://docs.pytorch.org/ao/main/workflows/qat.html)
- [Unsloth QAT Documentation]()