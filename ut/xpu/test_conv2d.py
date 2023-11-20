import torch
import torch.nn as nn
import intel_extension_for_pytorch
group=2
cpu_device="cpu"
dpcpp_device="xpu"

dtype=torch.float
x_cpu = torch.randn([1, 64, 256, 256], dtype=dtype, device=cpu_device, requires_grad=True)
grad_cpu = torch.full([1, 64, 256, 256], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
conv_cpu = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=group, bias=False)
y_cpu = conv_cpu(x_cpu)
y_cpu.backward(grad_cpu)
y_cpu_gw = conv_cpu.weight.grad.detach().clone()

conv_cpu.zero_grad()

x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_()
grad_dpcpp = grad_cpu.to(dpcpp_device)
conv_dpcpp = conv_cpu.to(dpcpp_device)
y_dpcpp = conv_dpcpp(x_dpcpp)
y_dpcpp.backward(grad_dpcpp)
y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

#self.assertEqual(y_cpu, y_dpcpp.cpu())
#self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)
