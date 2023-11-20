import torch
import intel_extension_for_pytorch

# out = beta * input + alpha * (mat1 @ mat2)
#########  test addmm_out  ########
print("=========FWD=======")
M_cpu = torch.randn(2, 3)
mat1_cpu = torch.randn(2, 3)
mat2_cpu = torch.randn(3, 3)
M = M_cpu.clone().to("xpu")
mat1 = mat1_cpu.clone().to("xpu")
mat2 = mat2_cpu.clone().to("xpu")
'''
print("M_before={}".format(M.cpu()))
out = torch.randn(M.shape).to("xpu")
torch.addmm(M, mat1, mat2, beta=2, alpha=1, out=M)
print("M_after={}".format(M.cpu()))
print("out={}".format(out.cpu()))
'''
M.addmm_(mat1, mat2)
M_cpu.addmm_(mat1_cpu, mat2_cpu)
print("M_xpu={}".format(M.cpu()))
print("M_cpu={}".format(M_cpu))
'''
#########  test addmm  ########
print("=========FWD=======")
M.requires_grad_()
mat1.requires_grad_()
mat2.requires_grad_()
print("M_before={}".format(M.cpu()))
out = torch.addmm(M, mat1, mat2, beta=3, alpha=2)
print("M_after={}".format(M.cpu()))
print("out={}".format(out.cpu()))

print("=========BWD=======")
grad = torch.randn(out.shape).to("xpu")
out.backward(grad)
'''

