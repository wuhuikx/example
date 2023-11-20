
import torch
import torch.nn as nn
import intel_extension_for_pytorch # noqa
dtype = torch.float
group = 1
'''
###########  2D  #########
deconv = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1, groups=group, bias=False)
deconv = deconv.to(memory_format=torch.channels_last)
x_cpu = torch.randn(2, 16, 128, 128, requires_grad=True).to(memory_format=torch.channels_last)
x_cpu.retain_grad()
gy_cpu = torch.randn(2, 32, 128, 128).to(memory_format=torch.channels_last)
y_cpu = deconv(x_cpu)
y_cpu.backward(gy_cpu)
gw_cpu = deconv.weight.grad.detach().clone()


x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_().to(memory_format=torch.channels_last)
deconv.zero_grad()
deconv = deconv.to('xpu')
deconv = deconv.to(memory_format=torch.channels_last)
gy_xpu = gy_cpu.to('xpu')
y_xpu = deconv(x_xpu)
y_xpu.backward(gy_xpu)
gw_xpu = deconv.weight.grad
diff = torch.max(torch.abs(y_cpu - y_xpu.cpu()))
print("---diff={}".format(diff))
diff_d = torch.max(torch.abs(x_cpu.grad - x_xpu.grad.cpu()))
print("---diff_d={}".format(diff_d))
diff_w = torch.max(torch.abs(gw_cpu - gw_xpu.cpu()))
print("---diff_w={}".format(diff_w))

'''
###########  2D  #########
deconv = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1, groups=group, bias=False)

x_cpu = torch.randn(2, 16, 128, 128, requires_grad=True)
x_cpu.retain_grad()
gy_cpu = torch.randn(2, 32, 128, 128)
y_cpu = deconv(x_cpu)
y_cpu.backward(gy_cpu)
gw_cpu = deconv.weight.grad.detach().clone()

with torch.xpu.onednn_layout():
    deconv.zero_grad()
    deconv = deconv.to('xpu')
    x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
    x_xpu.retain_grad()
    gy_xpu = gy_cpu.to('xpu')
    y_xpu = deconv(x_xpu)
    y_xpu.backward(gy_xpu)
    gw_xpu = deconv.weight.grad

diff = torch.max(torch.abs(y_cpu - y_xpu.cpu()))
print("---diff={}".format(diff))
diff_d = torch.max(torch.abs(x_cpu.grad - x_xpu.grad.cpu()))
print("---diff_d={}".format(diff_d))
diff_w = torch.max(torch.abs(gw_cpu - gw_xpu.cpu()))
print("---diff_w={}".format(diff_w))
#self.assertEqual(y_cpu, y_xpu.cpu())
#self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
#self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)

