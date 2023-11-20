
import torch
import torch.nn as nn
import intel_extension_for_pytorch # noqa
dtype = torch.float
group=2

###########  3D  #########
deconv = nn.ConvTranspose3d(16, 32, kernel_size=3, stride=1, padding=1, groups=group, bias=False)
deconv = deconv.to(memory_format=torch.channels_last_3d)
x_cpu = torch.randn(2, 16, 10, 128, 128, requires_grad=True).to(memory_format=torch.channels_last_3d)
x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
y_cpu = deconv(x_cpu)

deconv.zero_grad()
deconv = deconv.to('xpu')
y_xpu = deconv(x_xpu)
diff = torch.max(torch.abs(y_cpu - y_xpu.cpu()))
print("---diff={}".format(diff))


###########  3D  #########
deconv = nn.ConvTranspose3d(16, 32, kernel_size=3, stride=1, padding=1, groups=group, bias=False)
x_cpu = torch.randn(2, 16, 10, 128, 128, requires_grad=True)
x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
y_cpu = deconv(x_cpu)

deconv.zero_grad()
deconv = deconv.to('xpu')
y_xpu = deconv(x_xpu)
diff = torch.max(torch.abs(y_cpu - y_xpu.cpu()))
print("---diff={}".format(diff))
