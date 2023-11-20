import torch
from torch.nn.functional import interpolate
import intel_extension_for_pytorch

a = torch.randn(1, 3, 25, 26).to('xpu')
b = (24, 24)
for _ in range(10):
    c = interpolate(a, size=b, mode="area")

