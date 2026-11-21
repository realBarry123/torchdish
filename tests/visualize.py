from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from test_dish import BATCH, DishClassifier


MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "classifier.pt"
)


def _states_to_rgb(states):
    frame_count, channels, height, width = states.shape
    pixels = states.transpose(0, 2, 3, 1).reshape(-1, channels)
    centered = pixels - pixels.mean(axis=0, keepdims=True)

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


def visualize(model, image):
    if model.mutable.ndim != 4:
        raise ValueError("model.mutable must have shape (batch, channels, height, width)")

    image = torch.as_tensor(image)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("visualize expects a single image")
        image = image[0]
    if image.ndim == 3:
        if image.shape[0] != 1:
            raise ValueError("visualize expects a single-channel image")
        image = image[0]
    if image.ndim != 2 or min(image.shape) < 27:
        raise ValueError("image must be a single-channel image at least 27x27")

    device = model.mutable.device
    image = image[:27, :27].to(device=device, dtype=model.mutable.dtype)
    batch = image.unsqueeze(0).expand(model.mutable.shape[0], -1, -1)

    was_training = model.training
    previous_mutable = model.mutable
    previous_counter = model.counter
    previous_limit = model._limit
    snapshots = []

    try:
        model.eval()
        with torch.no_grad():
            model.reset()
            model.write(batch)
            snapshots.append(model.mutable[0].detach().cpu().numpy())

            for i in range(128):
                model.mutable = model.update(model.mutable)
                snapshots.append(model.mutable[0].detach().cpu().numpy())
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
        interval=100,
        repeat=False,
        blit=False,
    )


def visualize_immutable(model):
    if model.immutable.ndim != 4:
        raise ValueError(
            "model.immutable must have shape (batch, channels, height, width)"
        )

    immutable = model.immutable[:1].detach().cpu().numpy()
    rgb = _states_to_rgb(immutable)[0]

    figure, axis = plt.subplots()
    axis.imshow(rgb, interpolation="nearest")
    axis.set_title("Immutable state (PCA RGB)")
    axis.axis("off")
    return figure, axis


def main():
    from torchvision import datasets, transforms

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}; train the classifier first"
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    import __main__

    __main__.DishClassifier = DishClassifier
    model = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(model, DishClassifier):
        raise TypeError(f"Expected a saved DishClassifier at {MODEL_PATH}")
    model.eval()

    dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    animation = visualize(model, dataset[0][0])
    plt.show()
    visualize_immutable(model)
    plt.show()
    return animation


if __name__ == "__main__":
    main()
