import torch
from torch.testing import make_tensor 
import intel_extension_for_pytorch

device="xpu"
t = make_tensor([1] * 65, device, torch.float) 
torch.sum(t, dim=0)
