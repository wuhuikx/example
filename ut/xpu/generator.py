import torch

device = "xpu"
seed = 666
shape = [3, 4]

import intel_extension_for_pytorch
idx = torch.xpu.current_device()
generator = torch.xpu.default_generators[idx]
generator.manual_seed(seed)
xpu_input = torch.randn(shape, generator=generator, device="xpu")
print("xpu={}".format(xpu_input))

'''
generator = torch.Generator("cuda").manual_seed(seed)
cuda_input = torch.randn(shape, generator=generator, device="cuda")
print("cuda={}".format(cuda_input.cpu()))
'''

generator = torch.Generator("cpu").manual_seed(seed)
cpu_input = torch.randn(shape, generator=generator, device="cpu")
print("cpu={}".format(cpu_input))


