# Dish
*`class torchdish.Dish(mutable_shape: tuple, immutable_shape: tuple, train_immutable: bool=False)`*

Base class defining a cellular automaton with mutable and immutable states. Inherits `torch.nn.Module`. 

## State Variables

- `mutable` (`torch.Tensor`): the mutable state. 

- `immutable` (`torch.nn.Parameter`): the immutable state (set `train_immutable=True` to train) that wraps `torch.nn.Parameter`. 

- `train_immutable` (`bool`): toggles gradient tracking for `immutable`.

- `counter` (`int`): number of steps elapsed since the last `reset`. 

- `steps` (`int`): the number of updates that run before the output is read. 

- `noise` (`int`): the maximum magnitude of noise added to `steps` when `training` is set to `True`. For each training forward pass, the number of updates that run is uniformly distributed between `steps - noise` and `steps + noise`. 

## Methods

Subclasses should implement the `write`, `read`, and `update` methods. 

- `write(input: any)`: writes a module input to `mutable`. 

- `read() -> any`: reads a module output from `mutable`. 

- `update(x: tensor) -> torch.Tensor`: cellular automaton update function defined on the entire mutable state. Update may encorporate information from `immutable`. 

Other methods: 

- `reset()`: resets `counter` and `mutable`, and sets the step limit for the next run. 

- `stop() -> bool`: stopping condition, by default dependent on the value of `counter`. 

- `forward(x: torch.Tensor) -> torch.Tensor`: defines a full operation performed by the dish. 

## Example
```python
import torchdish as td
import torch
from torch import nn
from einops import repeat
from tqdm import tqdm

BATCH = 8
CHANNELS = 3

class MNISTClassifier(td.Dish):

    def __init__(self):
        super().__init__(
            mutable_shape=(1, CHANNELS, 28, 28),
            immutable_shape=(1, 3, 28, 28),
            train_immutable=True
        )
        self.conv1 = nn.Conv2d(
            in_channels=CHANNELS + self.immutable.shape[1], 
            out_channels=CHANNELS, 
            kernel_size=(3, 3), 
            padding="same"
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=CHANNELS, 
            out_channels=CHANNELS, 
            kernel_size=(3, 3), 
            padding="same"
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.steps = 32
        self.noise = 4

    def write(self, x):
        x = repeat(x, "b h w -> b c h w", c=CHANNELS)
        self.mutable = x

    def read(self):
        return self.mutable[:, :, 14, :10].mean(dim=1)

    def update(self, x):
        imm = self.immutable.expand(x.shape[0], -1, -1, -1) # expand immutable to fit batch
        r = torch.cat((x, imm), dim=1)
        r = self.conv1(r)
        r = self.relu(r)
        r = self.conv2(r)
        return x + r * 0.1
```