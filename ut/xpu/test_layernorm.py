import torch
import intel_extension_for_pytorch
import torch.nn as nn
from torch.autograd import Variable

layer_norm = nn.LayerNorm([1, 3, 3])
x_i = torch.randn([1, 1, 3, 3])

x_i[0][0][0][0] = 0.5021
x_i[0][0][0][1] = -0.9922 
x_i[0][0][0][2] = -0.7365
x_i[0][0][1][0] = 0.0629
x_i[0][0][1][1] = -2.0536
x_i[0][0][1][2] = -0.9989
x_i[0][0][2][0] = 0.4911
x_i[0][0][2][1] = 0.9744
x_i[0][0][2][2] = -1.9760

x_dpcpp_i = x_i.to("xpu")
x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
layer_norm_dpcpp = layer_norm.to("xpu")


x_dpcpp = x_dpcpp.to(torch.bfloat16)
y_dpcpp = layer_norm_dpcpp(x_dpcpp)

x_dpcpp = x_dpcpp.to(torch.float)
y_dpcpp = layer_norm_dpcpp(x_dpcpp)


#x_cpu = Variable(x_i, requires_grad=True)
#y_cpu = layer_norm(x_cpu)


