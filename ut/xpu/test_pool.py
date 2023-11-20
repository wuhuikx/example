import torch
import torch.nn as nn
import intel_extension_for_pytorch
dtype=torch.short
x = torch.randn([30, 40, 50])
x=x.to(torch.short).to("xpu")
m = nn.AdaptiveAvgPool2d((2, 2))
res=m(x)
