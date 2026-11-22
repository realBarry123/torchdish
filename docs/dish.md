# Dish
*`class torchdish.Dish(shape: tuple, immutable_shape: tuple, train_immutable: bool=False)`*

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