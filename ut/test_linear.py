import torch
import torch.nn.functional as F
import numpy as np
import intel_extension_for_pytorch

dtype=torch.bfloat16
input_shape = [256, 512, 1024]
linear = torch.nn.Linear(1024, 1024)

#input_shape = [8192, 8192]
#linear = torch.nn.Linear(8192, 8192)

linear=linear.to("xpu").to(dtype)
profiling=True
for step in range(3):
    with torch.autograd.profiler_legacy.profile(profiling, use_xpu=True, record_shapes=True) as prof:
    # with torch.autograd.profiler.emit_itt():
        # torch.profiler.itt.range_push('step_{}'.format(step))
        input = torch.randn(input_shape)
        input_xpu = input.to("xpu").bfloat16()
        res_xpu = linear(input_xpu)
        res_cpu = res_xpu.to("cpu")
        # torch.profiler.itt.range_pop()
    if profiling:
        torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profile.pt')
        torch.save(prof.table(sort_by="id", row_limit=-1), 'profile_detailed.pt')
        prof.export_chrome_trace('./profile_trace.json')

