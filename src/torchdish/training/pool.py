import torch


class Pool:
    def __init__(self, size, init_fn):
        # init_fn() returns one state of shape (1, *shape)
        self.init_fn = init_fn
        self.states = torch.cat([init_fn() for _ in range(size)], dim=0)

    def __len__(self):
        return self.states.shape[0]

    def sample(self, n):
        idx = torch.randperm(len(self), device=self.states.device)[:n]
        return self.states[idx].clone(), idx

    def update(self, idx, states):
        self.states[idx] = states.detach().to(self.states.device)

    def reset(self, idx=None):
        if idx is None:
            idx = torch.arange(len(self), device=self.states.device)
        self.states[idx] = torch.cat([self.init_fn() for _ in range(len(idx))]).to(self.states.device)

    def to(self, device):
        self.states = self.states.to(device)
        return self