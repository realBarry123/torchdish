import torch

@torch.no_grad()
def measure_receptive_field(model, pos: tuple[int], steps=None, eps=1e-3):
    steps = steps or model.steps
    model.eval()
    base = model.mutable.clone()
    perturbed = base.clone()
    perturbed[:, :, pos[0], pos[1]] += eps
    a, b = base, perturbed
    maps = []
    for _ in range(steps):
        a, b = model.update(a), model.update(b)
        maps.append((a - b).abs().sum(dim=(0, 1))) # (H, W)
    return torch.stack(maps) # (steps, H, W)