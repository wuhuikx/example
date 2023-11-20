# from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

from torch.quantization.quantize_jit import (convert_jit, prepare_jit)
from torch.jit._recursive import wrap_cpp_module

import pytest

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class Conv2dBinaryMul(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryMul, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        # return self.conv(x)
        return torch.mul(self.conv(x), a)
    
x = torch.randn([1, 64, 512, 512], device=cpu_device)
# a1 = torch.randn([1, 64, 512, 512], device=cpu_device)
a1 = torch.randn([1, 64, 512, 512], device=cpu_device)
a3 = a1.clone().to(dpcpp_device)
model = Conv2dBinaryMul(64, 64, kernel_size=3, stride=1, padding=(1, 1), bias=True)
y = model(x, a1)
# print("raw: ", y)

x = x.to("xpu")
model.to("xpu")
modelJit = torch.jit.script(model)
with torch.no_grad():
    y_dpcpp = modelJit(x, a3)
    print(modelJit.graph_for(x, a3))
    # print("fusion:", y_dpcpp.cpu())

