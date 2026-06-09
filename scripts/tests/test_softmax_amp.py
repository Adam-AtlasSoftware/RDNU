import torch
import torch.nn.functional as F

b = 8
head = 4
c_q = 8
h_w = 96 * 96

def test_grad(q1_scale=1.0):
    q1 = torch.randn(b, head, c_q, h_w, dtype=torch.float16, device='cuda') * q1_scale
    k = torch.randn(b, head, h_w, c_q, dtype=torch.float16, device='cuda') * q1_scale
    v = torch.randn(b, head, c_q, h_w, dtype=torch.float16, device='cuda') * q1_scale

    q1.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    with torch.cuda.amp.autocast(enabled=False):
        q1_fp32 = q1.to(torch.float32)
        k_fp32 = k.to(torch.float32)
        v_fp32 = v.to(torch.float32)
        
        # Add epsilon to normalize to prevent div by zero
        q1_n = F.normalize(q1_fp32, p=2, dim=-1, eps=1e-12)
        k_n  = F.normalize(k_fp32, p=2, dim=-2, eps=1e-12)

        attn = (q1_n @ k_n)
        
        attn_prob = F.softmax(attn, dim=-1)
        
        out = attn_prob @ v_fp32
        
    loss = out.sum()
        
    loss.backward()
    if torch.isnan(q1.grad).any().item():
        print(f"NaN in q1.grad! (q1_scale={q1_scale})")
    else:
        print(f"OK (q1_scale={q1_scale})")

for scale in [1e-6, 1e-3, 1.0, 1e3, 1e4]:
    test_grad(scale)
