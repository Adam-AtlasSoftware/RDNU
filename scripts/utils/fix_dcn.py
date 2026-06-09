import torch
import torch.nn.functional as F

class MockDCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 16
        self.kernel_size = (1, 7)
        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)
        self.weight = torch.nn.Parameter(torch.randn(16, 1, 1, 7))
        self.bias = torch.nn.Parameter(torch.randn(16))
        self.register_buffer('offset', torch.randn(1, 32, 1, 1))
        
    def forward(self, input):
        B, C, H, W = input.size()
        pad_y = self.kernel_size[0] // 2
        pad_x = self.kernel_size[1] // 2
        
        x_pad = F.pad(input, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0)
        
        shifted_x = torch.empty_like(input)
        for i in range(self.in_channels):
            dy = int(self.offset[0, 2 * i + 0, 0, 0].item())
            dx = int(self.offset[0, 2 * i + 1, 0, 0].item())
            
            y_start = pad_y + dy
            x_start = pad_x + dx
            shifted_x[:, i, :, :] = x_pad[:, i, y_start:y_start+H, x_start:x_start+W]
            
        return F.conv2d(shifted_x, self.weight, self.bias, stride=self.stride,
                        padding=self.padding, dilation=self.dilation)

m = MockDCN()
input = torch.randn(2, 16, 384, 384)
# Try torch.compile
try:
    m_c = torch.compile(m, mode="reduce-overhead")
    m_c(input)
    print("Success")
except Exception as e:
    print(e)
