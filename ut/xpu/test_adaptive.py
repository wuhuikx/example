import torch
import torch.nn as nn
import intel_extension_for_pytorch

#device = torch.device("xpu")
input = torch.randn(1, 3, 512, 512).to_mkldnn()
output_size = [308, 308]
torch._C._nn.adaptive_avg_pool2d(input, output_size)
'''
m = nn.AdaptiveAvgPool2d(7)
#input = torch.randn(1, 64, 10, 9)
input = torch.randn(1, 64, 10, 9).to_mkldnn()
#input = torch.randn(1, 64, 10, 9).to("mkldnn")
output = m(input)
'''
