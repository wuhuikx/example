import torch
import torch.nn.functional as F
import numpy as np
import intel_extension_for_pytorch

dtype=torch.float16
input_shape = [384, 1024]
linear1 = torch.nn.Linear(1024, 1024)
linear2 = torch.nn.Linear(1024, 1024)
linear3 = torch.nn.Linear(1024, 1024)


linear1=linear1.to("xpu").to(dtype)
linear2=linear2.to("xpu").to(dtype)
linear3=linear3.to("xpu").to(dtype)
profiling=True
for step in range(3):
    with torch.autograd.profiler_legacy.profile(profiling, use_xpu=True, record_shapes=True) as prof:
    # with torch.autograd.profiler.emit_itt():
        # torch.profiler.itt.range_push('step_{}'.format(step))
        input = torch.randn(1*1024*1024*1024)
        input = torch.randn(input_shape)
        input_xpu = input.to("xpu").to(dtype)
        res_xpu1 = linear1(input_xpu)
        res_xpu2 = linear2(input_xpu)
        res_xpu3 = linear3(input_xpu)
    if profiling:
        torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profile.pt')
        torch.save(prof.table(sort_by="id", row_limit=-1), 'profile_detailed.pt')
        prof.export_chrome_trace('./profile_trace.json')

