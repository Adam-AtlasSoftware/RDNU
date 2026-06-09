import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv

class CycleFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        self.register_buffer('offset', offset)

    def forward_dcn(self, x):
        B, C, H, W = x.size()
        return deform_conv2d_tv(x, self.offset.expand(B, -1, H, W), self.weight, self.bias)

    def forward_shift(self, x):
        B, C, H, W = x.size()
        # Max shift is kernel_size // 2
        pad_y = self.kernel_size[0] // 2
        pad_x = self.kernel_size[1] // 2
        
        # Pad the input
        x_pad = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0)
        
        shifted_x = torch.empty_like(x)
        for i in range(self.in_channels):
            dy = int(self.offset[0, 2 * i + 0, 0, 0].item())
            dx = int(self.offset[0, 2 * i + 1, 0, 0].item())
            
            # The original position in x_pad is y+pad_y, x+pad_x
            # We want to read from y+pad_y+dy, x+pad_x+dx
            y_start = pad_y + dy
            x_start = pad_x + dx
            shifted_x[:, i, :, :] = x_pad[:, i, y_start:y_start+H, x_start:x_start+W]
            
        return F.conv2d(shifted_x, self.weight, self.bias)

if __name__ == "__main__":
    x = torch.randn(2, 16, 24, 24)
    model = CycleFC(16, 16, (1, 7))
    out1 = model.forward_dcn(x)
    out2 = model.forward_shift(x)
    print("Difference:", (out1 - out2).abs().max().item())
