import torch
import random
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mnist_classify import MNISTClassifier
import torchdish as td

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = MNISTClassifier()
model.load_state_dict(torch.load(
    "tests/models/classifier_32s_3c+3_5x5o.pt",
    map_location=device,
    weights_only=False,
))
model.eval()

dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

td.plot_immutable(model)
plt.show()

while True: 
    sample = dataset[random.randint(0, 99)][0]
    animation = td.plot_mutable(model, sample, framerate=200)
    axis = animation._fig.axes[0]
    for row in range(2):
        for column in range(5):
            axis.add_patch(
                Rectangle(
                    (column * 5 - 0.5, row * 5 - 0.5),
                    5,
                    5,
                    fill=False,
                    edgecolor="yellow",
                    linewidth=1.5,
                )
            )
    prediction = torch.argmax(model(sample)).item()
    print("prediction:", prediction)
    plt.show()