from tqdm import tqdm
from einops import repeat, rearrange

import torch
import torch.nn.functional as F
from torch import nn

import torchdish as td

BATCH = 8
CHANNELS = 3

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

class Transposer(td.Dish):

    def __init__(self):
        super().__init__(
            mutable_shape=(1, CHANNELS, 24, 32),
            immutable_shape=(1, 3, 24, 32),
            train_immutable=True
        )
        self.conv1 = nn.Conv2d(
            in_channels=CHANNELS + self.immutable.shape[1], 
            out_channels=CHANNELS, 
            kernel_size=(3, 3), 
            padding="same"
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=CHANNELS, 
            out_channels=CHANNELS, 
            kernel_size=(3, 3), 
            padding="same"
        )
        #nn.init.zeros_(self.conv2.weight)
        #nn.init.zeros_(self.conv2.bias)

        self.steps = 128
        self.noise = 8

    def write(self, x):
        x = repeat(x, "b h w -> b 3 h w")
        self.mutable[:, :, 8:16, 8:16] = x

    def read(self):
        return self.mutable[:, :, 8:16, 16:24].mean(dim=1, keepdim=False)

    def update(self, x):
        r = x
        imm = self.immutable.expand(x.shape[0], -1, -1, -1)   # (1,K,H,W) -> (B,K,H,W)
        r = torch.cat((r, imm), dim=1)
        r = self.conv1(r)
        r = self.relu(r)
        r = self.conv2(r)
        return x + r * 0.1

if __name__ == "__main__":
    EPOCHS = 12
    EPOCH_SIZE = 1000
    K = 4

    model = Transposer().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(EPOCHS):
        total_loss = 0
        batch_count = 0
        for step in tqdm(range(EPOCH_SIZE)):
            x = torch.rand(BATCH, 8, 8).to(DEVICE)
            xT_hat = model(x)
            xT = x.permute((0, 2, 1))
            loss = F.mse_loss(xT_hat, xT)
            total_loss += loss.item()
            batch_count += 1

            optim.zero_grad(); loss.backward(); optim.step()
        print(total_loss / batch_count)
        torch.save(model.state_dict(), "tests/models/transposer.pt")

    import matplotlib.pyplot as plt
    anim = td.plot_mutable(model, torch.rand(1, 8, 8), framerate=50)
    plt.show()
