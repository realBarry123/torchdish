import torch
import torch.nn.functional as F
from torch import nn


class SobelFilter(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "kernel_x",
            torch.tensor(
                [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
            ).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "kernel_y",
            torch.tensor(
                [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
            ).unsqueeze(0),
            persistent=False,
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("SobelFilter expects input with shape (B, C, H, W)")

        channels = x.shape[1]
        if channels == 0:
            raise ValueError("SobelFilter requires at least one input channel")

        kernel_x = self.kernel_x.to(dtype=x.dtype, device=x.device)
        kernel_y = self.kernel_y.to(dtype=x.dtype, device=x.device)
        gradient_x = F.conv2d(x, kernel_x.expand(channels, -1, -1, -1), padding=1, groups=channels)
        gradient_y = F.conv2d(x, kernel_y.expand(channels, -1, -1, -1), padding=1, groups=channels)
        return torch.cat((gradient_x, gradient_y), dim=1)