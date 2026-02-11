import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import H,W,C

def perception(x):
    device = x.device

    identity = torch.tensor([[0,0,0],[0,1,0],[0,0,0]], dtype=torch.float32)
    sobel_x  = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)/8.0
    sobel_y  = sobel_x.T
    laplace  = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)/8.0

    kernels = torch.stack([identity, sobel_x, sobel_y, laplace])
    kernels = kernels[:, None, :, :].to(device)

    kernels = kernels.repeat(C, 1, 1, 1)

    return F.conv2d(x, kernels, padding=1, groups=C)

def alive_mask(x, alpha_channel=3, threshold=0.1):
    """
    x: (B, C, H, W)
    returns: (B, 1, H, W) float mask
    """
    alpha = x[:, alpha_channel:alpha_channel+1]  # (B, 1, H, W)

    alive = F.max_pool2d(
        alpha,
        kernel_size=3,
        stride=1,
        padding=1
    ) > threshold

    return alive.float()

def seed(batch_size=1, device="cpu"):
    x = torch.zeros(batch_size, C, H, W, device=device)
    x[:, 0:4, H//2, W//2] = 1.0  # single alive pixel
    return x

class NCA(nn.Module):
    def __init__(self, channels=C, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

        # Start near zero → stable training
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x, fire_rate=0.5):
        y = perception(x)
        dx = self.net(y)

        # stochastic update mask
        update_mask = (torch.rand_like(x[:, :1]) <= fire_rate).float()
        x = x + dx * update_mask
        x = x * alive_mask(x)
        return x
