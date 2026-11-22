import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from mnist_classify import MNISTClassifier
import torchdish as td

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = torch.load(
    "classifier.pt",
    map_location=device,
    weights_only=False,
)
model.eval()

dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)
animation = td.visualize(model, dataset[3][0])
plt.show()
# visualize_immutable(model)
plt.show()