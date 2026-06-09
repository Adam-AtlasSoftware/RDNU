import torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

device = torch.device('cuda')
model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()
transforms = Raft_Small_Weights.DEFAULT.transforms()

img1 = torch.rand(2, 3, 256, 256).to(device)
img2 = torch.rand(2, 3, 256, 256).to(device)

img1_t, img2_t = transforms(img1, img2)
print("Transforms applied:", img1_t.shape, img1_t.min(), img1_t.max())

with torch.no_grad():
    flow = model(img1_t, img2_t)[-1]
    
print("Flow shape:", flow.shape)
print("Flow min/max:", flow.min(), flow.max())
