import torch
import intel_extension_for_pytorch

dtype=torch.double
x = torch.randn(3, 2).to(dtype).to("xpu")
res = torch.where(x > 0, x, 0)
#torch.where(x > 0, x, 0.)
