import torch
import numpy as np
from .config import H,W

early_growth_state = torch.tensor([
    -0.0423,
    1.2397,
    0.0100,
    1.0570,
    -2.5695,
    -0.9226,
    -1.9094,
    1.0602,
    0.7348,
    0.4263,
    0.9746,
    0.2997,
    -1.5481,
    -1.6363,
    0.5318,
    -0.9766,
])

def circle_mask(cx, cy, r):
      y, x = np.ogrid[:H, :W]
      """Boolean mask for a circle centered at (cx, cy)."""
      return (x - cx)**2 + (y - cy)**2 < r**2

def rollout(
    model,
    x0,
    steps=100,
    ablate_channel=None,
    width_cut=None,
    killEye=None,
    prune=False,
    realEyeReplace=None,
    realEyeCapture=0,
    fakeEyeReplace=None,
    inhibit=None,
    injectionChannel=None,
    injectionTime=-1,
    injectionConcoction = False,
    earlyGrowthInject = False
):
    states = []
    x = x0.clone()
    oldEye = None

    for t in range(steps):
        if ablate_channel is not None:
            x[:, ablate_channel] *= 0.0
        if prune:
            x[:, :, ~circle_mask(H//2,W//2,18)] = 0.0
        if width_cut and t == width_cut:
            x[:, :, :, :W//2] = 0.0
        if killEye and t == killEye:
            x[:, :, circle_mask(36,28,6)] = 0.0
        if t == realEyeCapture:
            oldEye = x[:, :, circle_mask(36,28,6)]
        if realEyeReplace and t == realEyeReplace:
            x[0, :, circle_mask(36,28,6)] = oldEye
        if fakeEyeReplace and t>= fakeEyeReplace:
            x[0, :, circle_mask(36,28,5)] = 0.0
            x[0, 3, circle_mask(36,28,5)] = 1.0
        if injectionChannel is not None and (injectionTime <= t and t <= injectionTime+200):
            x[:, injectionChannel, circle_mask(36,28,6)] = 0.0
        if injectionConcoction and (injectionTime <= t and t <= injectionTime+100):
            x[:, 13, circle_mask(36,28,6)] = -1.0
            x[:, 6, circle_mask(36,28,6)] = 0.6
            x[:, 14, circle_mask(36,28,6)] = 0.7
            x[:, 12, circle_mask(36,28,6)] = -0.3
        if earlyGrowthInject and (injectionTime <= t and t <= injectionTime+100):
            mask = circle_mask(36,28,6)
            x[:, :, mask] = early_growth_state.view(1,16,1).expand(-1,-1,mask.sum())
            
        states.append(x.detach().cpu())

        if inhibit and t > 200:
            x = x + inhibit * (model(x) - x)  
        else:
            x = model(x)
        
    states = torch.stack(states)  # [T, B, C, H, W]
    return states[:, 0] #remove batch dim, returns [T, C, H, W]
