import torch
from torch import nn
from einops import rearrange

class FixedAttention2d(nn.Module):
    def __init__(self, channels, heads, H, W, r=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        ys, xs = ys.flatten(), xs.flatten()
        dy = (ys[:, None] - ys[None, :]).abs()
        dx = (xs[:, None] - xs[None, :]).abs()
        self.register_buffer("mask", ~((dy <= r) & (dx <= r)), persistent=False)

    def forward(self, x):
        B, C, H, W = x.shape
        t = rearrange(x, "b c h w -> b (h w) c")
        out, _ = self.attn(t, t, t, attn_mask=self.mask, need_weights=False)
        return rearrange(out, "b (h w) c -> b c h w", h=H, w=W)