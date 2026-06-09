import torch

def test():
    print("Testing FFT with AMP")
    x = torch.rand(2, 3, 384, 384, device='cuda')
    y = torch.rand(2, 3, 384, 384, device='cuda')
    
    with torch.cuda.amp.autocast():
        # Inside autocast, x and y might be float16
        print("x dtype inside autocast:", x.dtype)
        # Force float32
        x_fp32 = x.to(torch.float32)
        y_fp32 = y.to(torch.float32)
        print("x_fp32 dtype:", x_fp32.dtype)
        
        # In PyTorch, if autocast is enabled, torch.fft functions MIGHT automatically cast their inputs
        # back to float16 if they are registered in the autocast dispatch!
        
        out = torch.fft.rfft2(x_fp32)
        print("Output shape:", out.shape)
        
test()
