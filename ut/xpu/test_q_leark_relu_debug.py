import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch


dtype_inputs = torch.qint8
scale = 0.04

x_cpu = torch.randn([1, 1, 3, 4], device=torch.device("cpu"))
x_gpu = x_cpu.to('xpu')
Xelu = nn.LeakyReLU(0.1, inplace=True)

q_cpu = torch.quantize_per_tensor(x_cpu, scale, 0, dtype_inputs)
y_cpu = Xelu(q_cpu)

Xelu.to("xpu")
q_gpu = torch.quantize_per_tensor(x_gpu, scale, 0, dtype_inputs)
y_gpu = Xelu(q_gpu)
