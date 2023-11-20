import torch
# import intel_extension_for_ipex
import torch.nn.functional as F

device="cpu"
x = torch.randn((10, 3), device=device)
t = torch.empty(10, dtype=torch.int64, device=device).random_(0, 3) 
weight = torch.randn(1, 3, device=device)

F.nll_loss(x, t, weight=weight) 
