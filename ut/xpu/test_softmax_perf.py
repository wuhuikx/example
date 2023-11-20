import torch
import torch.nn.functional as F
import numpy as np
import intel_extension_for_pytorch

input_shape = [[16*16, 512, 512], [1, 8192, 8192], [1, 8192, 8732], [1, 8192, 30522], [128, 81, 8732], [1024, 128, 1000]]

for i in range(10):
    for j in range(1):
        size = input_shape[j]
        scale = 4
        if size[2] < 513:
            scale = 2
        memory = size[0] * size[1] * size[2] * 4 * scale * 1000 / 1024 / 1024 / 1024;
        print("----size={}, memory={}, dim={}".format(size, memory, 1))
        with torch.autograd.profiler_legacy.profile(True, use_xpu=True) as prof:
            a = torch.randn(size)
            a_xpu = a.to("xpu").bfloat16()
            res_xpu = F.softmax(a_xpu, 1)
            res_cpu = res_xpu.to("cpu")
            #diff1 = torch.max(torch.abs(res_xpu.cpu() - res))
        if i == 9:
             print(prof.key_averages().table(sort_by="self_xpu_time_total"))

        print("----size={}, memory={}, dim={}".format(size, memory, 2))
        with torch.autograd.profiler_legacy.profile(True, use_xpu=True) as prof:
            a = torch.randn(size)
            a_xpu = a.to("xpu").bfloat16()
            res_xpu = F.softmax(a_xpu, 2)
            res_cpu = res_xpu.to("cpu")
            #diff1 = torch.max(torch.abs(res_xpu.cpu() - res))
        if i == 9:
             print(prof.key_averages().table(sort_by="self_xpu_time_total"))
