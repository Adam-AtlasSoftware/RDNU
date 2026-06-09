import torch
import torch.nn.functional as F
import cv2
import numpy as np

def contrast_adaptive_sharpening(img, amount=0.8):
    # img: [B, C, H, W] in [0, 1]
    N = F.pad(img, (0, 0, 1, 0))[:, :, :-1, :]
    S = F.pad(img, (0, 0, 0, 1))[:, :, 1:, :]
    W = F.pad(img, (1, 0, 0, 0))[:, :, :, :-1]
    E = F.pad(img, (0, 1, 0, 0))[:, :, :, 1:]
    C = img

    min_val = torch.min(torch.min(torch.min(N, S), W), E)
    min_val = torch.min(min_val, C)
    
    max_val = torch.max(torch.max(torch.max(N, S), W), E)
    max_val = torch.max(max_val, C)
    
    d = torch.min(min_val, 1.0 - max_val)
    amp = torch.sqrt(torch.clamp(d / (max_val + 1e-6), min=0.0, max=1.0))
    
    # Sharpness parameter: from -0.125 (low) to -0.2 (high)
    w = amp * (-0.1 - (0.1 * amount))
    
    out = (N + S + W + E) * w + C
    out = out / (1.0 + 4.0 * w)
    
    return torch.clamp(out, 0.0, 1.0)

# Dummy test
t = torch.rand(1, 3, 256, 256)
out = contrast_adaptive_sharpening(t)
print("CAS success:", out.shape)
