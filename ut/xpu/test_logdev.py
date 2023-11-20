import torch
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

A=torch.tensor([[-0.8575, -0.0671,  0.9464],[ 0.4587, -0.2494, -1.3797], [ 1.2990,  0.5928,  1.0503]])
a = torch.det(A)
print("torch.det(A)", a.to(cpu_device))

A_dpcpp = A.to(dpcpp_device)
with torch.autograd.profiler.profile(False, use_xpu=True, with_calling_stack=True) as prof:
    a_dpcpp = torch.det(A_dpcpp)
#print(prof.key_averages().table(sort_by="self_xpu_time_total"))

print("torch.det(A_dpcpp)", a_dpcpp.to(cpu_device))
