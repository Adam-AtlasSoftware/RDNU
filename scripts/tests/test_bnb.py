import torch
import bitsandbytes as bnb

model = torch.nn.Linear(10, 10).cuda()
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-3)
x = torch.randn(10, 10).cuda()
y = model(x)
y.sum().backward()
optimizer.step()
print("BNB ok")
