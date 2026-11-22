# Training

## Pool

*`class torchdish.training.Pool(size: int, init_fn: Callable)`*

Sample pool for training as introduced by [Mordvintsev et al. 2020](https://distill.pub/2020/growing-ca/). The pool stores a batch of mutable states between training iterations.

`init_fn` must return a tensor with leading batch dimension `1`. All returned tensors must have the same shape, datatype, and device.

### Methods

- `len(pool) -> int`: number of stored states.

- `sample(n: int) -> tuple[torch.Tensor, torch.Tensor]`: samples up to `n` states without replacement and returns `(states, indices)`. `states` is a copy with shape `(n, *state_shape)`; `indices` identifies the corresponding entries in the pool and is on the pool's device.

- `update(indices: torch.Tensor, states: torch.Tensor)`: replaces the selected entries with `states`. 

- `reset(indices: torch.Tensor | list | None = None)`: reinitializes the selected entries. If `indices` is `None`, reinitializes the entire pool.

- `to(device) -> Pool`: moves the stored states to `device` and returns the pool.

### Example

```python
import torch
import torchdish as td

pool = td.training.Pool(
    size=128,
    init_fn=lambda: torch.zeros(1, 16, 32, 32),
)

states, indices = pool.sample(8)
# Run states through a model or training step.
pool.update(indices, states)
```

## LabelledPool

*`class torchdish.training.LabelledPool(size: int, init_fn: Callable)`*

`LabelledPool` is currently incomplete. Use `Pool` for now. 
