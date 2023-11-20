import torch
import intel_extension_for_pytorch

device="xpu"
dtype = torch.bfloat16
'''
b1 = torch.randn(1, 23, 15, device=device).to(dtype)
b2 = torch.randn(1, 15, 0, device=device).to(dtype)
res = torch.bmm(b1, b2)
print("---res={}".format(res.cpu()))
'''

b1 = torch.randn(10, 23, 15, device=device).to(dtype)
b2 = torch.randn(10, 15, 12, device=device).to(dtype)
b2=b2.permute(2,1,0).contiguous().permute(2,1,0)
print("---b2={}".format(b2.size()))
print("---b2={}".format(b2.stride()))
res = torch.bmm(b1, b2)
print("---res={}".format(res.cpu()))

