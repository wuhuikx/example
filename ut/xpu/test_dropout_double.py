import torch
import intel_extension_for_pytorch

a=torch.tensor([-3.0521,  0.8753, -8.0068, -4.5104, -1.4212]).double()
grad=torch.randn([5]).double()

a_xpu = a.clone().xpu()
grad_xpu = grad.clone().xpu()

a.requires_grad_(True)
a_xpu.requires_grad_(True)

m=torch.nn.Dropout()
out=m(a)
out.backward(grad)

out_xpu=m(a_xpu)
out_xpu.backward(grad_xpu)

print("a={}".format(a))
diff1=torch.max(torch.abs(out_xpu.cpu() - out))
diff2=torch.max(torch.abs(a_xpu.grad.cpu() - a.grad))
print("---output_xpu={}".format(out_xpu.cpu()))
print("---grad_xpu={}".format(a_xpu.grad.cpu()))
print("---out={}".format(out))
print("---grad={}".format(a.grad))
print("---diff1={}".format(diff1))
print("---diff2={}".format(diff2))
