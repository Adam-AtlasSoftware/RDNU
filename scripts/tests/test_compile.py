import torch

model = torch.nn.Linear(10, 10).cuda()
model = torch.compile(model, mode="reduce-overhead")
x = torch.randn(10, 10).cuda()
y = model(x)
y.sum().backward()
print("Compile ok")
