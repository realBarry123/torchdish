import random
import torch
from torch import nn

class Dish(nn.Module):

    def __init__(self, shape: tuple, immutable_shape, auto_reset=True, train_immutable=False):
        super().__init__()
        if len(shape) < 3:
            raise ValueError()
        self.register_buffer("mutable", torch.zeros(shape))
        self._train_immutable = train_immutable
        self._immutable = nn.Parameter(torch.zeros(immutable_shape), requires_grad=train_immutable)
        self.counter = 0
        # Whether to automatically reset between calls
        # Usually True unless you are trying some sequential nonsense
        self.auto_reset = auto_reset 
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
        return {"counter": self.counter, "steps": self.steps}

    def set_extra_state(self, state):
        self.counter = state["counter"]
        self.steps = state["steps"]

    def write(self, x: torch.Tensor): raise NotImplementedError()
    def read(self): raise NotImplementedError()

    def reset(self):
        self.counter = 0
        self._limit = self.steps
        if self.training: 
            self._limit += random.randint(-self.noise, self.noise)
        self.mutable = torch.zeros_like(self.mutable)

    def stop(self):
        # Default stopping condition
        return self.counter >= self._limit

    def update(self, x): raise NotImplementedError()

    def forward(self, x):
        if self.auto_reset:
            self.reset()
        self.write(x)
        while not self.stop():
            self.mutable = self.update(self.mutable)
        return self.read()
