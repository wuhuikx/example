import torch
import intel_extension_for_pytorch

a=torch.tensor(
    [[-3.2563, -6.1149, -7.3756, -5.2041, -7.8657],
    [-2.0768, -7.4830, -0.2323,  4.2791,  8.8506],
    [-4.8340, -0.3944,  6.5161,  8.1114, -8.5595],
    [ 7.7595,  5.4538, -6.9726, -8.7220, -8.9141],
    [ 2.4042, -8.1479,  7.6886,  4.1104, -3.3057]]).double()
grad=torch.randn([5, 5]).double()
print("a={}".format(a))
print("grad={}".format(grad))

a_xpu = a.clone().xpu()
grad_xpu = grad.clone().xpu()
grad.requires_grad_()
grad_xpu.requires_grad_()
a.requires_grad_()
a_xpu.requires_grad_()

print("------fwd_cpu-----")
b=torch.linalg.det(a)
print("------bwd_cpu-----")
b.backward()

print("------fwd_xpu-----")
b_xpu=torch.linalg.det(a_xpu)
print("------bwd_xpu-----")
b_xpu.backward()
print("------end-----")

diff1=torch.max(torch.abs(b_xpu.cpu() - b))
diff2=torch.max(torch.abs(a_xpu.grad.cpu() - a.grad))
print("b={}".format(b))
print("b_xpu={}".format(b_xpu))
print("a_grad={}".format(a.grad))
print("a_grad_xpu={}".format(a_xpu.grad.cpu()))
print("---diff1={}".format(diff1))
print("---diff2={}".format(diff2))
