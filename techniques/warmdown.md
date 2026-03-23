# Warmdown Learning-Rate Scheduling

Warmdown is a late-stage learning-rate decay that activates once training nears its allotted time budget or final iterations. Instead of predefining a static decay schedule, warmdown continuously monitors either elapsed wall-clock time or progress through the iteration budget and linearly reduces the LR so that optimization tapers off smoothly before the run must stop.

## Why use warmdown?
- **Deadline compliance:** Ensures a run wraps up cleanly when constrained by strict wall-clock limits (e.g., benchmarking competitions, shared cluster slots).
- **Stability:** Shrinking the LR toward zero at the end reduces the chance of large parameter jumps right before serialization.
- **Consistency:** All optimizer parameter groups follow the same decay factor, keeping relative step sizes intact.

## How it works
1. **Track elapsed time or iterations.** Maintain `elapsed_ms` since training began and, if applicable, the maximum allowed runtime.
2. **Compute remaining budget.** When `remaining_ms <= warmdown_window_ms`, set `scale = remaining_ms / warmdown_window_ms`. If remaining time is larger, keep `scale = 1.0`.
3. **Apply to optimizers.** Multiply each param group’s base LR by `scale` every step. Gradually driving `scale` to zero forces all LRs to vanish exactly when the budget expires.
4. **Optional iteration-based variant.** Replace wall-clock math with `(total_iters - step)` when time budgets aren’t available.

## Practical PyTorch example
The script below trains a toy model with a 5‑second budget. Once the remaining time drops below 1.5 seconds, warmdown linearly decays the learning rate toward zero.

```python
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
base_lrs = [group["lr"] for group in optimizer.param_groups]

max_seconds = 5.0
warmdown_window = 1.5  # seconds
start_time = time.perf_counter()
step = 0

while True:
    step += 1
    elapsed = time.perf_counter() - start_time
    remaining = max(max_seconds - elapsed, 0.0)
    if remaining <= warmdown_window:
        scale = remaining / warmdown_window if warmdown_window > 0 else 0.0
    else:
        scale = 1.0
    for (group, base_lr) in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * scale

    x = torch.randn(128, 8, device=device)
    y = torch.randint(0, 2, (128,), device=device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

    if remaining <= 0.0:
        break

print(f"Completed {step} steps. Final LR: {optimizer.param_groups[0]['lr']:.2e}")
```

Running the script shows the learning rate automatically shrinking to zero as the clock approaches the 5‑second budget, ensuring the optimizer settles before termination.
