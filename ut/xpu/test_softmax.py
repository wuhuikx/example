import torch
import torch.nn.functional as F
import numpy as np
import intel_extension_for_pytorch

#input_shape = [[16*16, 512, 512], [1, 8192, 8192], [1, 8192, 8732], [1, 8192, 30522], [128, 81, 8732], [1024, 128, 1000]]
input_shape = [[16*256, 512, 512]]

for step in range(3):
    for j in range(1):
        size = input_shape[j]
        scale = 2
        memory = size[0] * size[1] * size[2] * 4 * scale * 1000 / 1024 / 1024 / 1024;
        with torch.autograd.profiler.emit_itt():
            torch.profiler.itt.range_push('step_{}'.format(step))
            a = torch.randn(size)
            a_xpu = a.to("xpu").bfloat16()
            res_xpu = F.softmax(a_xpu, 2)
            res_cpu = res_xpu.to("cpu")
            torch.xpu.synchronize()
            torch.profiler.itt.range_pop()

