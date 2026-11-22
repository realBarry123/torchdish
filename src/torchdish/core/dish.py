import random
import torch
from torch import nn
from einops import repeat

class Dish(nn.Module):

    def __init__(self, shape: tuple, immutable_shape, train_immutable=False):
        super().__init__()
        if len(shape) < 3:
            raise ValueError("Not enough mutable dimensions")
        self.register_buffer("mutable", torch.zeros(shape), persistent=False)
        self._train_immutable = train_immutable
        self._immutable = nn.Parameter(torch.zeros(immutable_shape), requires_grad=train_immutable)
        self.counter = 0
        self.steps = 32
        self.noise = 4
        self.reset()

    @property
    def immutable(self):
        return self._immutable
    @immutable.setter
    def immutable(self, value):
        with torch.no_grad():
            self._immutable.copy_(value)

    @property
    def train_immutable(self):
        return self._train_immutable
    @train_immutable.setter
    def train_immutable(self, value: bool):
        self._train_immutable = value
        self._immutable.requires_grad_(self._train_immutable)

    def get_extra_state(self):
        return {
            "counter": self.counter, 
            "steps": self.steps, 
            "noise": self.noise
        }

    def set_extra_state(self, state):
        self.counter = state["counter"]
        self.steps = state["steps"]
        self.noise = state["noise"]

    def write(self, x: torch.Tensor): raise NotImplementedError()
    def read(self): raise NotImplementedError()

    def reset(self):
        self.counter = 0
        self._limit = self.steps
        if self.training: 
            self._limit += random.randint(-self.noise, self.noise)
        self.mutable = torch.zeros([1,] + list(self.mutable.shape[1:])).to(self.immutable.device)

    def stop(self):
        # Default stopping condition
        return self.counter >= self._limit

    def update(self, x): raise NotImplementedError()

    def forward(self, x):
        self.reset()
        self.mutable = repeat(self.mutable, "1 c h w -> b c h w", b=x.shape[0]).contiguous()
        self.write(x)
        while not self.stop():
            self.mutable = self.update(self.mutable)
            self.counter += 1
        return self.read()
    
    def resume(self, steps):
        self.counter = 0
        self._limit = steps
        while not self.stop():
            self.mutable = self.update(self.mutable)
        return self.read()
