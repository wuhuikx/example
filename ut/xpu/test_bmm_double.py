import torch
import intel_extension_for_pytorch

a=torch.randn(3, 5, 5).double()
b=torch.randn(3, 5, 5).double()
grad=torch.randn(3, 5, 5).double()
a_xpu = a.clone().xpu()
b_xpu = b.clone().xpu()
grad_xpu=grad.clone().xpu()

a.requires_grad_(True)
b.requires_grad_(True)
a_xpu.requires_grad_(True)
b_xpu.requires_grad_(True)

c=torch.bmm(a,b)
c.backward(grad)
c_xpu=torch.bmm(a_xpu, b_xpu)
c_xpu.backward(grad_xpu)

diff1 = torch.max(torch.abs(c_xpu.cpu() - c))
diff2 = torch.max(torch.abs(a_xpu.grad.cpu() - a.grad))
print("---diff1={}".format(diff1))
print("---diff2={}".format(diff2))
