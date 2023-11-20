import torch
import intel_extension_for_pytorch
import math
device="cpu"
# dtype=torch.bfloat16
dtype=torch.float

m1=1
n1=1
k1=0

M = torch.randn(n1, m1, device=device).to(dtype)          
m = torch.randn(n1, k1, device=device).to(dtype) 
v = torch.randn(k1, m1, device=device).to(dtype)     
print("---M={}".format(M.cpu()))
print("---m={}".format(m.cpu()))
print("---v={}".format(v.cpu()))
#M = M_cpu.to(dtype).to(device)
#m = m_cpu.to(dtype).to(device)
#v = v_cpu.to(dtype).to(device)

numpy_dtype = dtype
if dtype in {torch.bfloat16}:
    numpy_dtype = torch.float

#alpha = 0 
alpha = 1.2
beta = 0.8
'''
print("M={}".format(M.cpu()))
print("m={}".format(m.cpu()))
print("v={}".format(v.cpu()))
res1 = torch.addmm(M, m, v, alpha=alpha, beta=beta)
print("---res1={}".format(res1.cpu()))

res3 = alpha * (m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy())
if beta != 0:
    res3 += (beta * M).to(numpy_dtype).cpu().numpy()
res3 = torch.from_numpy(res3).to(dtype)
print("---res3={}".format(res3.cpu()))

diff_res3 = torch.max(torch.abs(res3.cpu()-res1.cpu()))
print("---diff_res3={}".format(diff_res3))
'''

res2 = torch.full_like(M, math.nan)
res2 = torch.addmm(M, m, v, alpha=alpha, out=res2)
#res2 = torch.addmm(M, m, v, alpha=alpha, beta=beta)
#print("--res1={}".format(res1))
print("--res2={}".format(res2.cpu()))

