import copy
import pytest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device('cpu')
xpu_device = torch.device("xpu")

dtype=torch.bfloat16

seq=5
rnn = nn.GRU(379, 681, num_layers=3, batch_first=True)
rnn_xpu = copy.deepcopy(rnn).to("xpu").to(dtype)
input = torch.randn(512, seq, 379)
h0 = torch.randn(3, 512, 681)
input_xpu = input.to("xpu").to(dtype)
h0_xpu = h0.to("xpu").to(dtype)
grad_output = torch.randn(512, seq, 681)
grad_output_xpu = grad_output.to("xpu").to(dtype)

input.requires_grad = True
h0.requires_grad = True
output, hn = rnn(input, h0)

grad_output.requires_grad = True
output.backward(grad_output)

param_grad = []
for param in rnn._parameters.values():
    param_grad.append(param._grad.clone())

input_xpu.requires_grad = True
h0_xpu.requires_grad = True
output_xpu, hn_xpu = rnn_xpu(input_xpu, h0_xpu)

grad_output_xpu.requires_grad = True
output_xpu.backward(grad_output_xpu)
 
param_grad_xpu = []
for param in rnn_xpu._parameters.values():
    param_grad_xpu.append(param._grad.clone())

diff = torch.max(torch.abs(output - output_xpu.cpu()))
print("----diff={}".format(diff))
'''
self.assertEqual(output, output_xpu.cpu())
self.assertEqual(h0, h0.cpu())
self.assertEqual(input.grad, input_xpu.grad.cpu())
self.assertEqual(h0.grad, h0_xpu.grad.cpu())
for i in range(len(param_grad)):
    self.assertEqual(param_grad[i], param_grad_xpu[i].cpu())
'''
