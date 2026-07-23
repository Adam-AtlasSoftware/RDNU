import torch
import numpy as np
import os

def export_to_hlsl_bin(pth_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading weights from {pth_path}...")
    
    # Load checkpoint to CPU
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=True)
    
    # Extract state_dict (handles both raw state_dicts and wrapped checkpoint dicts)
    state_dict = checkpoint.get('params', checkpoint)
    
    exported_count = 0
    
    for key, tensor in state_dict.items():
        # Clean up distributed training prefixes if they exist
        clean_key = key.replace('module.', '')
        
        # Convert tensor to FP16 numpy array
        np_tensor = tensor.detach().half().numpy()
        
        # PyTorch Conv2D layout: [Out_C, In_C, H, W]
        # HLSL / DX12 optimal layout: [Out_C, H, W, In_C]
        if len(np_tensor.shape) == 4:
            np_tensor = np_tensor.transpose(0, 2, 3, 1)
            
        # Ensure array is C-contiguous in memory after transposition
        np_tensor = np.ascontiguousarray(np_tensor)
        
        # Flatten and save to binary file
        file_path = os.path.join(output_dir, f"{clean_key}.bin")
        np_tensor.tofile(file_path)
        
        print(f"Exported: {clean_key} | HLSL Shape: {np_tensor.shape}")
        exported_count += 1

    print(f"\nSuccessfully exported {exported_count} FP16 weight tensors to '{output_dir}/'")

if __name__ == "__main__":
    # Replace with the path to the final model (e.g., net_g_300000.pth)
    PTH_FILE = "net_g_latest.pth" 
    OUT_DIR = "rdg_hlsl_weights"
    
    export_to_hlsl_bin(PTH_FILE, OUT_DIR)