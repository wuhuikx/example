import torch
import torch.nn as nn
import intel_extension_for_pytorch
dtype = torch.float

conv_cpu = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
conv_dpcpp = conv_cpu.to("xpu") 

x_cpu = torch.randn([1, 64, 256, 256], dtype=dtype, device="cpu", requires_grad=True)
grad_cpu = torch.full([1, 64, 256, 256], 1e-3, dtype=dtype, device="cpu", requires_grad=True)

x_dpcpp = x_cpu.to("xpu").requires_grad_()
grad_dpcpp = grad_cpu.to("xpu")
y_dpcpp = conv_dpcpp(x_dpcpp)
y_dpcpp.backward(grad_dpcpp) 
