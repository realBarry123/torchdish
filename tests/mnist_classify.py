import torchdish as td
import torch
from torch import nn
from torch.utils.data import DataLoader
from einops import repeat, rearrange
from tqdm import tqdm

BATCH = 8
CHANNELS = 3


class MNISTClassifier(td.Dish):

    def __init__(self):
        super().__init__(
            shape=(BATCH, CHANNELS, 32, 32),
            immutable_shape=(BATCH, 3, 32, 32),
            train_immutable=True
        )
        self.conv1 = nn.Conv2d(
            in_channels=CHANNELS + self.immutable.shape[1], 
            out_channels=CHANNELS, 
            kernel_size=(3, 3), 
            padding="same"
        )
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.conv2 = nn.Conv2d(
            in_channels=CHANNELS, 
            out_channels=CHANNELS, 
            kernel_size=(3, 3), 
            padding="same"
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.steps = 32
        self.noise = 0

    def write(self, x):
        # x: (B, H, W)
        x = repeat(x, "b h w -> b c h w", c=CHANNELS)
        self.mutable[:, :, :27, :27] = x

    def read(self):
        # return self.mutable[:, :, 14, :10].mean(dim=1)
        logits = []
        for i in range(2):
            for j in range(5):
                logit_area = self.mutable[:, :3, 5*i: 5*i+5, 5*j: 5*j + 5]
                logit = logit_area.mean((2, 3), keepdim=False) # neighbourhood mean
                logit = logit.mean(1, keepdim=True) # mean across channels
                logits.append(logit)
        return torch.cat(logits, dim=1)
        

    def update(self, x):
        r = x
        r = torch.cat((r, self.immutable), dim=1)
        r = self.conv1(r)
        r = self.relu(r)
        r = self.conv2(r)
        return x + r * 0.1


def _prepare_images(images):
    if images.ndim == 4 and images.shape[1] == 1:
        images = images.squeeze(1)
    images = images[..., :27, :27]
    if images.shape[-2:] != (27, 27):
        raise ValueError("MNISTClassifier requires images at least 27x27")
    if images.shape[0] != BATCH:
        raise ValueError(
            f"MNISTClassifier requires batches of {BATCH}; "
            "set DataLoader(batch_size=8, drop_last=True)"
        )
    return images


def evaluate_classifier(model, eval_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in eval_loader:
            images = _prepare_images(images).to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

    if total == 0:
        raise ValueError("eval_loader yielded no full classifier batches")
    return correct / total


def train_classifier(model, train_loader, eval_loader, epochs=1, lr=1e-3):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for images, labels in tqdm(train_loader):
            images = _prepare_images(images).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = ce_loss(logits, labels.long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        if batch_count == 0:
            raise ValueError("train_loader yielded no full classifier batches")
        epoch_loss = total_loss / batch_count
        accuracy = evaluate_classifier(model, eval_loader, device)
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(accuracy)
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"loss={epoch_loss:.4f}, test_accuracy={accuracy:.2%}"
        )
        torch.save(model.state_dict(), "tests/models/classifier.pt")

    return epoch_losses, epoch_accuracies


if __name__ == "__main__":
    from torchvision import datasets, transforms

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
    )
    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH,
        shuffle=False,
        drop_last=True,
    )

    model = MNISTClassifier()
    train_classifier(model, train_loader, test_loader, epochs=12, lr=0.001)

