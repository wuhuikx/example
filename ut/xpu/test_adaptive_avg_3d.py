import torch
import intel_extension_for_pytorch
import torch.nn as nn

dpcpp_device = "xpu"

x = torch.randn([10, 16, 30, 40, 50])
grad = torch.randn([10, 16, 2, 2, 2])
conv_cpu1 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
pool_cpu = nn.AdaptiveAvgPool3d((2, 2, 2))
conv_cpu2 = nn.Conv3d(16, 16, kernel_size=1, stride=1, bias=False)

# 5D contiguous input
# CPU
input_cpu = x.clone()
input_cpu.requires_grad_(True)
grad_cpu = grad.clone()
output_cpu = conv_cpu2(pool_cpu(conv_cpu1(input_cpu)))
output_cpu.backward(grad_cpu)

conv_cpu1.zero_grad()
conv_cpu2.zero_grad()

# XPU
with torch.xpu.onednn_layout():
    input_xpu = x.clone().to(dpcpp_device)
    input_xpu.requires_grad_(True)
    grad_xpu = grad.clone().to(dpcpp_device)
    conv_dpcpp1 = conv_cpu1.to(dpcpp_device)
    pool_dpcpp = pool_cpu.to(dpcpp_device)
    conv_dpcpp2 = conv_cpu2.to(dpcpp_device)
    output_xpu = conv_dpcpp2(pool_dpcpp(conv_dpcpp1(input_xpu)))
    output_xpu.backward(grad_xpu)

diff = torch.max(torch.abs(output_cpu - output_xpu.cpu()))
print("diff={}".format(diff))
diff = torch.max(torch.abs(input_cpu.grad - input_xpu.grad.cpu()))
print("diff={}".format(diff))
