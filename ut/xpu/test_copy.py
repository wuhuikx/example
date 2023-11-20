import torch
import intel_extension_for_pytorch

a=torch.rand(23, 72, 72, 256, 2).float().xpu()
b=torch.rand(1, 1024, 1024, 1024).float().xpu()

for i in range(5):
    with torch.autograd.profiler_legacy.profile(True, use_xpu=True) as prof:
        # b = b+1
        c = a.clone()
    print(str(prof.key_averages().table(sort_by="self_cpu_time_total")))
