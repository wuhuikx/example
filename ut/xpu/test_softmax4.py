import torch
import torch.nn.functional as F
import numpy as np
import intel_extension_for_pytorch


dim = 1
dtypes = [torch.bfloat16, torch.float]
dtypes = [torch.bfloat16]

for index, dtype in enumerate(dtypes):
    input_shape = np.array([[1, 2, 3], [16, 5, 17], [7, 7, 355], [256, 32, 1023], [256, 11, 255], [16*16, 512, 512], [16*16, 512, 513], [1, 8192, 8192], [1, 8192, 8732], [1, 8192, 30522], [128, 81, 8732], [1024, 128, 1000]])
    #input_shape = np.array([[1, 8192, 8732]])
    iteration = 1
    # accuracy
    for i in range(iteration):
        for j in range(input_shape.shape[0]):
            size = input_shape[j]
            print("acc, dtype={}, size={}, dim={}".format(dtype, size, dim))
            a = torch.randn(tuple(size))
            grad = torch.randn(tuple(size))

            a_cpu = a.clone()
            a_cpu.requires_grad_()
            grad_cpu = grad.clone()
            res_cpu = F.softmax(a_cpu, dim)
            res_cpu.backward(grad_cpu)

            a_xpu = a.clone().to("xpu").to(dtype)
            a_xpu.requires_grad_()
            grad_xpu = grad.clone().to("xpu").to(dtype)
            res_xpu = F.softmax(a_xpu, dim)
            res_xpu.backward(grad_xpu)

            diff = torch.max(torch.abs(res_xpu.cpu() - res_cpu))
            diff_grad = torch.max(torch.abs(a_xpu.grad.cpu() - a_cpu.grad))
            print("---diff={}".format(diff))
            print("---diff_grad={}".format(diff_grad))
