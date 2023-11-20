import torch
import intel_extension_for_pytorch
import torch.nn as nn

dtype = torch.float
cpu_device = "cpu"
dpcpp_device = "xpu"


x = torch.randn([2, 16, 10, 128, 128], dtype=dtype, device=cpu_device)
grad = torch.full([2, 16, 2, 2, 2], 1e-3, dtype=dtype, device=cpu_device)

x_cpu = x.clone()
x_cpu.requires_grad_(True)
grad_cpu = grad.clone()

x_dpcpp = x.clone().to(dpcpp_device)
x_dpcpp.requires_grad_(True)
#grad_dpcpp = grad.clone().to(dpcpp_device)
grad_dpcpp = torch.full([2, 16, 2, 2, 2], 1e-3, dtype=dtype, device=dpcpp_device)

conv_cpu1 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
avg_pool_cpu = nn.AdaptiveAvgPool3d((2, 2, 2))
#avg_pool_cpu = nn.AdaptiveMaxPool3d((2, 2, 2))
conv_cpu2 = nn.Conv3d(32, 16, kernel_size=1, bias=False)

y_cpu = conv_cpu2(avg_pool_cpu(conv_cpu1(x_cpu)))
y_cpu.backward(grad_cpu)

conv_dpcpp1 = conv_cpu1.to(dpcpp_device)
avg_pool_dpcpp = nn.AdaptiveAvgPool3d((2, 2, 2))
# avg_pool_dpcpp = avg_pool_cpu.to(dpcpp_device)
conv_dpcpp2 = conv_cpu2.to(dpcpp_device)


with torch.no_grad():
    with torch.xpu.onednn_layout():
        y_dpcpp = conv_dpcpp2(avg_pool_dpcpp(conv_dpcpp1(x_dpcpp)))
        #print("y_dpcpp={}".format(y_dpcpp.cpu()))
        y_dpcpp.backward(grad_dpcpp)

#diff = torch.max(torch.abs(y_cpu - y_dpcpp.cpu()))
#print("-diff={}".format(diff))
diff = torch.max(torch.abs(x_cpu.grad - x_dpcpp.grad.cpu()))
print("-diff={}".format(diff))
    # self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
