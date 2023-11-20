import torch
import torch.nn as nn
import torch.nn.functional as F

import intel_extension_for_pytorch
torch._C._jit_set_profiling_mode(False)                     
torch._C._jit_set_profiling_executor(False) 

dtype=torch.float
cpu_device = "cpu"
dpcpp_device = "xpu"

class PadConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PadConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        #return F.relu(x, inplace=False)
        return x


x = torch.randn([1, 2, 3, 5], device=cpu_device)
print("x={}".format(x.shape))

model = PadConv2d(2, 2, kernel_size=1, stride=1, padding=(1, 2), bias=True)
y = model(x)
print("raw: ", y)
print("raw: ", y.shape)

x = x.to("xpu")
model.to("xpu")

modelJit = torch.jit.script(model)
modelJit.to("xpu")
print(modelJit.graph_for(x))

with torch.no_grad():
    # print(modelJit.graph_for(x, a2))
    with torch.autograd.profiler_legacy.profile(True, use_xpu=True) as prof:
        y_dpcpp = modelJit(x)
    print(prof.table(sort_by="id", row_limit=-1))
    prof.export_chrome_trace('inference_profiling.json')
    print("fusion:", y_dpcpp.cpu())

print("fusion={}".format(y_dpcpp.shape))
#self.assertEqual(y, y_dpcpp.to(cpu_device))
#del modelJit


