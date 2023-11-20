import torch
import torch.nn as nn
import intel_extension_for_pytorch
dtype = torch.float
dpcpp_device = "xpu"
cpu_device = "cpu"

conv = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False, groups=2).to(dpcpp_device)
x = torch.randn([1, 256, 3, 3], dtype=torch.float, device=cpu_device, requires_grad=True).to(dpcpp_device)
grad = torch.full([1, 64, 3, 3], 1e-3, dtype=torch.float, device=cpu_device, requires_grad=True).to(dpcpp_device)
real = conv(x)
# real.backward(grad)
# y_dpcpp_gw = conv.weight.grad.detach().clone()

# conv.zero_grad()

# conv_cpu = conv.cpu()
# x_cpu = x.cpu()
# grad_cpu = grad.cpu()
# ref = conv_cpu(x_cpu)
# ref.backward(grad_cpu)
# y_cpu_gw = conv_cpu.weight.grad.detach().clone()



'''
    def test_group_conv_blk(self, dtype=torch.float):
        conv = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False, groups=2).to(cpu_device)
        x = torch.randn([1, 256, 3, 3], dtype=torch.float, device=cpu_device, requires_grad=True).to(cpu_device)
        grad = torch.full([1, 64, 3, 3], 1e-3, dtype=torch.float, device=cpu_device, requires_grad=True)
        ref = conv(x)
        ref.backward(grad)
        y_cpu_gw = conv.weight.grad.detach().clone()

        conv.zero_grad()

        with torch.xpu.onednn_layout():
            conv_xpu = conv.to(dpcpp_device)
            x_xpu = x.to(dpcpp_device)
            grad_xpu = grad.to(dpcpp_device)
            real = conv_xpu(x_xpu)
            real.backward(grad_xpu)
            y_dpcpp_gw = conv_xpu.weight.grad.detach().clone()

            self.assertEqual(real.cpu(), ref)
            self.assertEqual(y_dpcpp_gw.cpu(), y_cpu_gw)

    def test_group_conv_channels_last(self, dtype=torch.float):
        conv = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False, groups=2).to(cpu_device)
        x = torch.randn([1, 256, 3, 3], dtype=torch.float, device=cpu_device, requires_grad=True).to(cpu_device)
        grad = torch.full([1, 64, 3, 3], 1e-3, dtype=torch.float, device=cpu_device, requires_grad=True)
        ref = conv(x)
        ref.backward(grad)
        y_cpu_gw = conv.weight.grad.detach().clone()

        conv.zero_grad()

        conv_xpu = conv.to(dpcpp_device).to(memory_format=torch.channels_last)
        x_xpu = x.to(dpcpp_device).to(memory_format=torch.channels_last)
        grad_xpu = grad.to(dpcpp_device).to(memory_format=torch.channels_last)
        real = conv_xpu(x_xpu)
        real.backward(grad_xpu)
        y_dpcpp_gw = conv_xpu.weight.grad.detach().clone()

        self.assertEqual(real.cpu(), ref)
        self.assertEqual(y_dpcpp_gw.cpu(), y_cpu_gw)
        self.assertTrue(real.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(y_dpcpp_gw.is_contiguous(memory_format=torch.channels_last))
'''
