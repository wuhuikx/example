import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")
print_graph = True

class LinearReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearReLU, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear(x))
        return x

class LinearBinaryMul(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryMul, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, a):
        y = torch.mul(self.linear(x), a)
        return y

'''
shapes = [[[2], [2, 6]]]
#shapes = [[[4, 2], [2, 6]]]
#shapes = [[[3, 4, 2], [2, 6]]]
#shapes = [[[2, 3, 4, 2], [2, 6]]]
for shape in shapes:
    ic = shape[1][0]
    oc = shape[1][1]
    x = torch.randn(shape[0], device=cpu_device)
    print("---x={}".format(x.shape))
    print("---ic={}".format(ic))
    print("---oc={}".format(oc))

    model = LinearReLU(ic, oc)
    y = model(x)
    print("raw: ", y)

    x = x.to("xpu")
    model.to("xpu")
    modelJit = torch.jit.trace(model, x)

    with torch.no_grad():
        for i in range(3):
            if print_graph and i==2:
                print(modelJit.graph_for(x))
        y_dpcpp = modelJit(x)
        print("fusion:", y_dpcpp.cpu())
    del modelJit
    diff = torch.max(torch.abs(y - y_dpcpp.cpu()))
    print("-----cpu={}".format(y.shape))
    print("-----xpu={}".format(y_dpcpp.shape))
    print("-----diff={}".format(diff))
'''

linear_shapes = [[[4, 2], [2, 6], [4, 6]]]
for shape in linear_shapes:
    '''
    ic = shape[1][0]
    oc = shape[1][1]
    x = torch.randn(shape[0], device=cpu_device)
    x1 = torch.randn(shape[2], device=cpu_device)
    # x1 = torch.ones(shape[2], device=cpu_device) * 2

    model = LinearBinaryMul(ic, oc)
    '''
    x = torch.randn([2, 4], device=cpu_device)
    x1 = torch.randn([2, 4], device=cpu_device)
    model = LinearBinaryMul(4, 4)

    y = model(x, x1)
    print("raw: ", y)

    x_xpu = x.to("xpu")
    x1_xpu = x1.to("xpu")
    model.to("xpu")
    print("-----------ut0-----------")
    modelJit = torch.jit.script(model)

    with torch.no_grad():
        for i in range(3):
            print("-----------ut2-----------")
            if print_graph:
                print(modelJit.graph_for(x_xpu, x1_xpu))
            print("-----------ut2_2-----------")
        print("-----------ut3-----------")
        y_dpcpp = modelJit(x_xpu, x1_xpu)
        print("-----------ut4-----------")
        print("fusion:", y_dpcpp.cpu())
    del modelJit
    diff = torch.max(torch.abs(y - y_dpcpp.cpu()))
    print("-----cpu={}".format(y.shape))
    print("-----xpu={}".format(y_dpcpp.shape))
    print("-----diff={}".format(diff))

