import torch
import numpy as np
from einops import repeat, rearrange
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def _states_to_rgb(states):
    frame_count, channels, height, width = states.shape
    pixels = states.transpose(0, 2, 3, 1).reshape(-1, channels)
    centered = pixels - pixels.mean(axis=0, keepdims=True)

    if channels == 1: 
        print(states.shape)
        states = repeat(states, "f 1 h w -> f 3 h w")
        return rearrange(states, "f c h w -> f h w c")
    if channels == 3:
        print(states.shape)
        return rearrange(states, "f c h w -> f h w c")

    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    components = eigenvectors[:, np.argsort(eigenvalues)[::-1][:3]]
    projected = centered @ components
    if projected.shape[1] < 3:
        projected = np.repeat(projected[:, :1], 3, axis=1)

    rgb = projected.reshape(frame_count, height, width, 3)
    lower = rgb.min(axis=(0, 1, 2), keepdims=True)
    scale = rgb.max(axis=(0, 1, 2), keepdims=True) - lower
    rgb = np.divide(
        rgb - lower,
        scale,
        out=np.zeros_like(rgb),
        where=scale > 1e-8,
    )
    return np.clip(rgb, 0, 1)

def visualize(model, x):
    if model.mutable.ndim != 4:
        raise ValueError("model.mutable must have shape (batch, channels, height, width)")

    x = torch.as_tensor(x).to(device=model.mutable.device, dtype=model.mutable.dtype)
    # batch = x.unsqueeze(0).expand(model.mutable.shape[0], -1, -1)

    was_training = model.training
    previous_mutable = model.mutable
    previous_counter = model.counter
    previous_limit = model._limit
    snapshots = []

    try:
        model.eval()
        with torch.no_grad():
            model.reset()
            model.write(x)
            snapshots.append(model.mutable[0].detach().clone().cpu().numpy())

            for i in range(model.steps + model.noise):
                model.mutable = model.update(model.mutable)
                frame = model.mutable[0]
                frame = (frame - frame.min()) / (frame.max()-frame.min()) # added per-frame normalization
                snapshots.append(frame.detach().cpu().numpy())
    finally:
        model.mutable = previous_mutable
        model.counter = previous_counter
        model._limit = previous_limit
        model.train(was_training)


    frames = _states_to_rgb(np.stack(snapshots))
    figure, axis = plt.subplots()
    image_artist = axis.imshow(frames[0], interpolation="nearest")
    title = axis.set_title("Step 0")
    axis.axis("off")

    def update(frame_index):
        image_artist.set_data(frames[frame_index])
        title.set_text(f"Step {frame_index}")
        return image_artist, title

    return FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=50,
        repeat=False,
        blit=False,
    )
