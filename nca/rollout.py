import torch
import numpy as np
from .config import H,W

def circle_mask(cx, cy, r):
      y, x = np.ogrid[:H, :W]
      """Boolean mask for a circle centered at (cx, cy)."""
      return (x - cx)**2 + (y - cy)**2 < r**2

def rollout(model, x0, steps, ablate_channel=None, cut=None, killEye=False, trim=False):
    states = []
    x = x0.clone()
    oldEye = None
    oldShape = None

    for t in range(steps):
        if trim:
            x[:, :, ~circle_mask(H//2,W//2,18)] = 0.0
        if ablate_channel is not None:
            x[:, ablate_channel] *= 0.1
        if cut is not None and t == cut:
            x[:, :, :, :W//2] = 0.0
        if killEye and t == 101:
            oldEye = x[:, :, circle_mask(36,28,6)]
            x[:, :, circle_mask(36,28,6)] = 0.0
        if killEye and t >= 300:
            # x[0, :, circle_mask(36,28,6)] = oldEye

            # fake eye
            x[0, :, circle_mask(36,28,5)] = 0.0
            x[0, 3, circle_mask(36,28,5)] = 1.0
            x[0, 4, circle_mask(36,28,5)] = 1.0
            
        states.append(x.detach().cpu())
        x = model(x)
        
    states = torch.stack(states)  # [T, B, C, H, W]
    # print(oldEye.shape)
    return states[:, 0] #remove batch dim, returns [T, C, H, W]
