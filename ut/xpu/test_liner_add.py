import torch
import intel_extension_for_pytorch
import math

device="cpu"
dtype=torch.bfloat16
#dtype=torch.float

M_cpu = torch.randn(10, 25)
m_cpu = torch.randn(10, 50)
v_cpu = torch.randn(50, 25)
#m_cpu = torch.randn(10, 50)
#v_cpu = torch.randn(50, 25)
M = M_cpu.clone().to(dtype).to(device)
m = m_cpu.clone().to(dtype).to(device)
v = v_cpu.clone().to(dtype).to(device)

#print("---M_cpu={}".format(M_cpu))
numpy_dtype = dtype
if dtype in {torch.bfloat16}:
    numpy_dtype = torch.float
alpha = 1
#beta = 1
beta = 0.8

res3 = alpha * (m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy())
#res3_1 = res3 + (beta * (M.to(numpy_dtype).cpu())).numpy()
#res3_2 = res3 + (beta * M).to(numpy_dtype).cpu().numpy()
#res3_1 = beta * (M.cpu())
#res3_2 = (beta * M).cpu()
#res3_1 = beta * (M.to(numpy_dtype).cpu())

with torch.autograd.profiler_legacy.profile(True, use_xpu=True) as prof:
    res3_1 = res3 + (beta * (M.to(numpy_dtype).cpu())).numpy()
    res3_2 = res3 + (beta * M).to(numpy_dtype).cpu().numpy()
    res3_3 = res3 + (beta * (M.cpu().to(numpy_dtype).cpu())).numpy()
    res3_4 = res3 + (beta * M.cpu()).to(numpy_dtype).cpu().numpy()
#print(prof.table(sort_by="id", row_limit=100000))

res3_1 = torch.from_numpy(res3_1).to(dtype)
res3_2 = torch.from_numpy(res3_2).to(dtype)
res3_3 = torch.from_numpy(res3_3).to(dtype)
res3_4 = torch.from_numpy(res3_4).to(dtype)
diff2 = torch.max(torch.abs(res3_1 - res3_2))
print("---diff2={}".format(diff2))
diff2 = torch.max(torch.abs(res3_3 - res3_4))
print("---diff2={}".format(diff2))
#diff2 = torch.max(torch.abs(res3_1 - res3_3))
#print("---diff2={}".format(diff2))
#diff2 = torch.max(torch.abs(res3_2 - res3_))
#print("---diff2={}".format(diff2))

