import torch


class Pool:
    def __init__(self, size, init_fn):
        # init_fn() returns one state of shape (1, *shape)
        self.init_fn = init_fn
        self.states = torch.cat([init_fn() for _ in range(size)], dim=0)

    def __len__(self):
        return self.states.shape[0]

    def sample(self, n):
        idcs = torch.randperm(len(self), device=self.states.device)[:n]
        return self.states[idcs].clone(), idcs

    def update(self, idcs, states):
        self.states[idcs] = states.detach().to(self.states.device)

    def reset(self, idcs=None):
        if idcs is None:
            idcs = torch.arange(len(self), device=self.states.device)
        self.states[idcs] = torch.cat([self.init_fn() for _ in range(len(idcs))]).to(self.states.device)

    def to(self, device):
        self.states = self.states.to(device)
        return self


class LabelledPool(Pool):
    def __init__(self, size, init_fn):
        # init_fn() -> (state, label) with state (1, *shape), label (1,)
        super().__init__(size, init_fn)

    def sample(self, n):
        items, idcs = super().sample(n)
        states, labels = zip(*items)
        return torch.cat(states), torch.cat(labels), idcs

    def update(self, idx, states, labels):
        super().update(idx, list(zip(states.split(1), labels.split(1))))