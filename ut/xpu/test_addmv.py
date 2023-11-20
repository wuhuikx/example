import torch
import intel_extension_for_pytorch

device="xpu"
dtype = torch.half
#dtype = torch.bfloat16
numpy_dtype = dtype
if dtype in {torch.bfloat16}:
    numpy_dtype = torch.float

'''
b1 = torch.randn(1, 23, 15, device=device).to(dtype)
b2 = torch.randn(1, 15, 0, device=device).to(dtype)
res = torch.bmm(b1, b2)
print("---res={}".format(res.cpu()))

#b1 = torch.randn(10, 23, 15, device=device).to(dtype)
#b2 = torch.randn(10, 15, 12, device=device).to(dtype)
#b2=b2.permute(2,1,0).contiguous().permute(2,1,0)
b1 = torch.randn(50, 100, device=device).to(dtype)
b2 = torch.randn(100, 1, device=device).to(dtype)
#b2=b2.permute(2,1,0).contiguous().permute(2,1,0)
print("---b2={}".format(b2.size()))
print("---b2={}".format(b2.stride()))
res = torch.addmv(b1, b2)
print("---res={}".format(res.cpu()))
'''
alpha = 1
beta = 0
#beta = 0.8
ts = [
    0.2 * torch.randn(50, device=device).to(dtype),
    0.2 * torch.randn(1, device=device).to(dtype).expand(50),]

vs = [
    0.2 * torch.randn(100, device=device).to(dtype),
    0.2 * torch.randn(1, device=device).to(dtype).expand(100),]

ms = [
    0.2 * torch.ones((), device=device).to(dtype).expand(50, 100),
    0.2 * torch.randn((1,100), device=device).to(dtype).expand(50, 100),
    0.2 * torch.randint(3, (50, 1), dtype=torch.float, device=device).to(dtype).expand(50, 100),
    0.2 * torch.randn((50, 100), device=device).to(dtype),
    0.2 * torch.randn((100, 50), device=device).to(dtype).t(),
]

for i in range(1):
    for j in range(1):
        for k in range(1):
            i = 0
            j = 1
            k = 2
            print("====i={}".format(i))
            print("====j={}".format(j))
            print("====k={}".format(k))
            t=ts[i]
            v=vs[j]
            m=ms[k]
            print("---t={}".format(t.size()))
            print("---t={}".format(t.stride()))
            print("---m={}".format(m.size()))
            print("---m={}".format(m.stride()))
            print("---v={}".format(v.size()))
            print("---v={}".format(v.stride()))
            res1 = torch.addmv(t, m, v, alpha=alpha, beta=beta)

            res3 = alpha * (m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy())
            #if beta != 0:
            #res3_1 = res3 + (beta * (t.to(numpy_dtype).cpu())).numpy()         
            #res3_2 = res3 + (beta * t).to(numpy_dtype).cpu().numpy()         
            #res3_1 = torch.from_numpy(res3_1).to(dtype)     
            #res3_2 = torch.from_numpy(res3_2).to(dtype)                                     
            res3_1 = torch.from_numpy(res3).to(dtype)     
            res3_2 = torch.from_numpy(res3).to(dtype)                                     

            #diff1 = torch.max(torch.abs(res1.cpu()-res2.cpu()))   
            diff2 = torch.max(torch.abs(res1.cpu()-res3_1.cpu()))
            print("---diff2={}".format(diff2))                                 
            diff2 = torch.max(torch.abs(res1.cpu()-res3_2.cpu()))               
            print("---diff2={}".format(diff2))                             
