import time
import torch
import torch.nn.functional as F

b, t, c, h, w = 16, 6, 3, 384, 384
scale = 4

gt_img = torch.rand(b, t, c, h, w, device='cuda')
gt_Depth = torch.rand(b, t, 1, h, w, device='cuda')

def run_pytorch_jitter():
    t0 = time.perf_counter()
    from basicsr.data.gaming_degradations import halton_sequence
    thetas_t = []
    for frame_idx in range(t):
        jitter_x = halton_sequence(frame_idx + 1, 2) - 0.5
        jitter_y = halton_sequence(frame_idx + 1, 3) - 0.5
        shift_x = jitter_x * scale
        shift_y = jitter_y * scale
        theta = torch.tensor([[1.0, 0.0, -2.0 * shift_x / w],
                              [0.0, 1.0, -2.0 * shift_y / h]], dtype=gt_img.dtype, device=gt_img.device)
        thetas_t.append(theta)
        
    theta_flat = torch.stack(thetas_t, dim=0).unsqueeze(0).repeat(b, 1, 1, 1).view(b*t, 2, 3)
    
    gt_img_flat = gt_img.reshape(b*t, c, h, w)
    grid_flat = F.affine_grid(theta_flat, gt_img_flat.size(), align_corners=False)
    hr_jittered_flat = F.grid_sample(gt_img_flat, grid_flat, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    lq_img_flat = F.interpolate(hr_jittered_flat, size=(h // scale, w // scale), mode='bicubic', align_corners=False)
    lq_img_flat = torch.clamp(lq_img_flat, 0.0, 1.0)
    lq_Image = lq_img_flat.view(b, t, c, h // scale, w // scale)
    
    gt_Depth_flat = gt_Depth.reshape(b*t, 1, h, w)
    lq_dep_flat = F.interpolate(gt_Depth_flat, size=(h // scale, w // scale), mode='nearest')
    lq_Depth = lq_dep_flat.view(b, t, 1, h // scale, w // scale)
    torch.cuda.synchronize()
    return time.perf_counter() - t0

# warm up
for _ in range(5): run_pytorch_jitter()

times = [run_pytorch_jitter() for _ in range(50)]
print(f"PyTorch Jitter time per batch (16): {sum(times)/len(times):.4f}s")
