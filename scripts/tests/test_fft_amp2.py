import torch
import torch.nn as nn

class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred):
        pred_fp32 = pred.to(torch.float32)
        print("Inside forward: pred dtype:", pred.dtype, "pred_fp32 dtype:", pred_fp32.dtype)
        # Even with to(float32), autocast might intercept the torch.fft.rfft2 call 
        # and cast the inputs to float16 BEFORE calling the actual kernel!
        pred_fft = torch.fft.rfft2(pred_fp32)
        return pred_fft

loss = FFTLoss().cuda()
# What if we pass a float16 tensor?
x16 = torch.rand(2, 3, 384, 384, dtype=torch.float16, device='cuda')
with torch.cuda.amp.autocast():
    try:
        out = loss(x16)
        print("Success")
    except Exception as e:
        print("Failed:", e)
