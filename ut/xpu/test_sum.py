import torch
import ipex

a=torch.Tensor([1, 0, 0]).to(bool)
A=a.to("xpu")


#a_res = a.sum(-1, False, dtype=torch.float);
with torch.autograd.profiler.profile(True, use_xpu=True, with_calling_stack=True) as prof:
    A_res = A.sum(-1, False, dtype=torch.float);
print(prof)  

#print("--a_res={}".format(a_res));
print("--A_res={}".format(A_res.cpu()));

