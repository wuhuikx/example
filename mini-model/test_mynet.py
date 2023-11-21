import torch
import torch.nn as nn
import intel_extension_for_pytorch

class MyNet(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sequential = nn.Sequential(
            nn.Linear(3, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.sequential(x)


net = MyNet() 
net.to("xpu")
print(next(net.parameters()).device)
print(next(net.parameters()).dtype)

net.half()
print(next(net.parameters()).dtype)
