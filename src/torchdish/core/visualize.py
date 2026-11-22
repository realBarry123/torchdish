import torch
import numpy as np
from einops import repeat, rearrange
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

def _states_to_rgb(states):
    frame_count, channels, height, width = states.shape
    pixels = states.transpose(0, 2, 3, 1).reshape(-1, channels)
    centered = pixels - pixels.mean(axis=0, keepdims=True)

    if channels == 1: 
        states = repeat(states, "f 1 h w -> f 3 h w")
        return rearrange(states, "f c h w -> f h w c")
    if channels == 3:
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

def plot_mutable(model, x, framerate=100, normalize_mode="local"):
    if model.mutable.ndim != 4:
        raise ValueError("model.mutable must have shape (batch, channels, height, width)")
    if normalize_mode not in ("local", "global", "clip"):
        raise ValueError("normalize_mode not recognized, please use one of 'local', 'global', 'clip'")

    x = torch.as_tensor(x)
    if x.ndim == 4:
        if x.shape[1] != 1:
            raise ValueError("visualize expects single-channel image batches")
        x = x.squeeze(1)
    elif x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0).unsqueeze(0)
    elif x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError("image must have shape (H, W), (1, H, W), or (B, H, W)")
    x = x.to(device=model.mutable.device, dtype=model.mutable.dtype)

    was_training = model.training
    previous_mutable = model.mutable
    previous_counter = model.counter
    previous_limit = model._limit
    snapshots = []

    try:
        model.eval()
        with torch.no_grad():
            model.reset()
            if model.mutable.shape[0] != x.shape[0]:
                model.mutable = model.mutable[:1].expand(
                    x.shape[0], -1, -1, -1
                ).contiguous()
            model.write(x)
            snapshots.append(model.mutable[0].detach().clone().cpu().numpy())

            for i in range(model.steps + model.noise):
                model.mutable = model.update(model.mutable)
                frame = model.mutable[0]
                if normalize_mode == "local": 
                    lo, hi = np.percentile(frame, [1, 99]) 
                    frame = (frame - lo) / (hi - lo + 1e-6) # per frame norm
                snapshots.append(frame.detach().cpu().numpy())

            snaps = np.stack(snapshots)
            if normalize_mode == "global":
                lo, hi = np.percentile(snaps, [1, 99])
                snaps = ((snaps - lo) / (hi - lo + 1e-6))
            snaps = snaps.clip(0, 1)
    finally:
        model.mutable = previous_mutable
        model.counter = previous_counter
        model._limit = previous_limit
        model.train(was_training)


    frames = _states_to_rgb(snaps)
    figure, axis = plt.subplots()
    figure.subplots_adjust(bottom=0.22)
    image_artist = axis.imshow(frames[0], interpolation="nearest")
    title = axis.set_title("Step 0")
    axis.axis("off")

    slider_axis = figure.add_axes((0.15, 0.08, 0.62, 0.04))
    frame_slider = Slider(
        slider_axis,
        "Frame",
        0,
        len(frames) - 1,
        valinit=0,
        valstep=1,
    )
    button_axis = figure.add_axes((0.85, 0.05, 0.12, 0.09))
    play_button = Button(button_axis, "Stop")
    playback = {"frame": 0, "playing": True}

    def show_frame(frame_index):
        frame_index = int(frame_index)
        playback["frame"] = frame_index
        image_artist.set_data(frames[frame_index])
        title.set_text(f"Step {frame_index}")

    def update(frame_index):
        show_frame(frame_index)
        frame_slider.eventson = False
        frame_slider.set_val(frame_index)
        frame_slider.eventson = True
        return image_artist, title

    def on_slider_change(frame_index):
        show_frame(frame_index)
        figure.canvas.draw_idle()

    def frame_sequence():
        while True:
            yield playback["frame"]
            if playback["playing"]:
                playback["frame"] = (playback["frame"] + 1) % len(frames)

    def toggle_playback(_event):
        playback["playing"] = not playback["playing"]
        if playback["playing"]:
            play_button.label.set_text("Stop")
            animation.event_source.start()
        else:
            play_button.label.set_text("Resume")
            animation.event_source.stop()
        figure.canvas.draw_idle()

    frame_slider.on_changed(on_slider_change)
    play_button.on_clicked(toggle_playback)
    animation = FuncAnimation(
        figure,
        update,
        frames=frame_sequence,
        interval=framerate,
        blit=False,
        cache_frame_data=False,
    )
    animation._torchdish_widgets = (frame_slider, play_button)
    return animation

def plot_immutable(model, normalize_mode="scale"):
    if model.immutable.ndim != 4:
        raise ValueError(
            "model.immutable must have shape (batch, channels, height, width)"
        )
    if normalize_mode not in ("scale", "clip"):
        raise ValueError("normalize_mode not recognized, please use one of 'scale', 'clip'")

    immutable = model.immutable[:1].detach().cpu().numpy()
    if normalize_mode == "scale":
        lo, hi = np.percentile(immutable, [1, 99])
        immutable = ((immutable - lo) / (hi - lo + 1e-6))
    immutable = immutable.clip(0, 1)
    rgb = _states_to_rgb(immutable)[0]

    figure, axis = plt.subplots()
    axis.imshow(rgb, interpolation="nearest")
    axis.set_title("Immutable state")
    axis.axis("off")
    return figure, axis
