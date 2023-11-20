import torch
import intel_extension_for_pytorch

device="xpu"
x = torch.rand((1,), device=device).expand((6,))
src = torch.rand((3,), device=device) 
ind = torch.tensor([2, 1, 0], device=device, dtype=torch.int64)  

src.scatter_(0, ind, src) 
