import torch
import gc
import torch.nn as nn

class DummyRDGModel(nn.Module):
    """
    Simulates a heavy video rendering / restoration network pass 
    with feature extraction, recurrent alignment, and upscaling.
    """
    def __init__(self, in_channels=6, num_feat=16, scale=2):
        super().__init__()
        # 3D convolution to handle temporal frame processing (6 frames)
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, num_feat, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(num_feat, num_feat * 4, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Reconstruction layers
        self.reconstruct = nn.Sequential(
            nn.Conv3d(num_feat * 4, num_feat * 4, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Upsampling simulation step
        self.upsample = nn.ConvTranspose3d(num_feat * 4, 3, kernel_size=(1, 4, 4), stride=(1, scale, scale), padding=(0, 1, 1))

    def forward(self, x):
        # Permute from [B, F, C, H, W] to standard PyTorch format [B, C, F, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        feat = self.feature_extractor(x)
        out = self.reconstruct(feat)
        return self.upsample(out)

def profile_batch_scaling():
    print("Initializing BasicSR Video Model Memory Step-Profiler (With Activation & Gradient Tracking)...")
    if not torch.cuda.is_available():
        print("Error: CUDA not available.")
        return

    device = torch.device("cuda:0")
    
    frames = 6
    channels = 6  # Base 3 + motion + depth arrays
    height, width = 384, 384
    num_feat = 16
    scale = 2

    # Instantiate the model architecture and push to GPU
    print("Instantiating dummy network model on GPU...")
    model = DummyRDGModel(in_channels=channels, num_feat=num_feat, scale=scale).to(device)
    model.train() # Enable training mode to track activations

    # Test scaling boundaries
    for test_batch in [3, 4, 6, 8, 12, 16, 24, 32]:
        print(f"\nTesting Per-GPU Batch Size: {test_batch}...")
        
        # Clear VRAM cache completely before test
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        
        try:
            # 1. Input Generation (Simulating AMP / Float16)
            inputs = torch.randn(test_batch, frames, channels, height, width, dtype=torch.float16, device=device)
            allocated_input_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
            print(f"  -> Input Tensors VRAM: {allocated_input_mem:.2f} MB")
            
            # 2. Forward Pass Simulation (Using Automatic Mixed Precision)
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(inputs)
                # Targets match output shape, initialized as float16
                targets = torch.randn_like(outputs)
                loss = torch.mean((outputs - targets) ** 2)
            
            # 3. Backward Pass Simulation (Generates matching gradients)
            loss.backward()
            
            # 4. Measure Peak Footprint
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f"  -> Peak Estimated Pass VRAM: {peak_mem:.2f} MB / 24576 MB")
            
            if peak_mem > 21000:  # 85% safety threshold check
                print(f"  [!] Batch Size {test_batch} is approaching absolute safety limits.")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [X] Batch Size {test_batch} FAILED with CUDA Out of Memory.")
                break
            else:
                raise e

if __name__ == "__main__":
    profile_batch_scaling()
