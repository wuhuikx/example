import torch
import intel_extension_for_pytorch

device="xpu"

x = torch.rand((1,), device=device).expand((3,))
src = torch.rand((6,), device=device)
ind = torch.tensor([2, 1, 0], device=device, dtype=torch.int64)
#torch.gather(src, 0, ind, out=src)
print("---test1----")
a=ind.clone()
print("---test2----")
b=ind[1:]
print("---test3----")
c=ind[:1]
print("---test4----")
torch.gather(a, 0, b, out=c)
print("---test5----")
#torch.gather(ind.clone(), 0, ind[1:], out=ind[:1])
