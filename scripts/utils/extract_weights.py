import torch
import numpy as np
import os

def extract_weights(pth_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading weights from {pth_path}...")
    # Load the checkpoint (map to CPU to avoid VRAM usage)
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    # If the model was saved with DataParallel/DDP, keys will start with "module."
    state_dict = checkpoint.get('params', checkpoint)
    
    for key, tensor in state_dict.items():
        # Clean up key name for file system
        clean_key = key.replace('module.', '')
        
        # Convert to FP16 numpy array
        np_tensor = tensor.detach().half().numpy()
        
        # PyTorch Conv2d weight format: [OutChannels, InChannels, KernelH, KernelW]
        # For HLSL and DX12 (depending on your specific matrix multiplier layout), 
        # you may need to transpose this. For standard memory mapping, keeping it contiguous works.
        # But if you need [KernelH, KernelW, InChannels, OutChannels], uncomment the next line:
        # if len(np_tensor.shape) == 4:
        #     np_tensor = np_tensor.transpose(2, 3, 1, 0)
            
        file_path = os.path.join(output_dir, f"{clean_key}.bin")
        np_tensor.tofile(file_path)
        
        print(f"Saved {clean_key} -> {np_tensor.shape} (FP16)")

if __name__ == "__main__":
    # Example usage (point this to your actual .pth file)
    PTH_FILE = "RDG/experiments/train_RDG_s_x4_GameIR/models/net_g_7334.pth"
    OUT_DIR = "RDNU/exported_weights"
    extract_weights(PTH_FILE, OUT_DIR)
    print("Weight extraction complete!")
