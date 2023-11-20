import torch
import torch.nn.functional as F
import numpy as np
import intel_extension_for_pytorch


dim = 2
dtypes = [torch.bfloat16, torch.float]
dtypes = [torch.float]
#dtypes = [torch.bfloat16]
profiling = True 
for index, dtype in enumerate(dtypes):
    input_shape = np.array([[1, 2, 3], [16, 5, 17], [256, 32, 1023], [256, 11, 255], [16*16, 512, 512], [16*16, 512, 513], [1, 8192, 8192], [1, 8192, 8732], [1, 8192, 30522], [128, 81, 8732], [1024, 128, 1000]])
    #input_shape = np.array([[256,512,256], [1,512,256], [256,512,512], [1,512,512], [256,512,1024], [1,512,1024], [256,512,2048], [1,512,2048], [1,128,1000], [16,128,1000], [1, 8732, 8732], [1, 8000, 8000], [1,8732,30522], [1, 8732, 30000], [1,1,30522]])
    #input_shape = np.array([[256,512,512], [512, 384, 384], [1, 8192, 30522], [128, 81, 8732]])
    input_shape = np.array([[512, 3000, 21], [512,3000,91], [256, 512, 512], [512, 384, 384], [1, 8192, 30522], [1, 64, 8192], [1, 8192, 64], [28, 4096, 8], [2048, 64, 3], [1024, 128, 1000]])
    #input_shape = np.array([[512,3000,21], [512, 3000, 91]])
    # input_shape = np.array([[512, 3000, 21], [512,3000,91]])
    input_shape = np.array([[256, 512, 512]])
    # input_shape = np.array([[512, 384, 384]])
    #input_shape = np.array([[1, 64, 8192]])
    #input_shape = np.array([[1, 8192, 64]])
    #input_shape = np.array([[512,3000,21]])
    #input_shape = np.array([[28, 4096, 8]])
    #input_shape = np.array([[28, 4095, 8]])
    #input_shape = np.array([[16, 5, 17]])
    # input_shape = np.array([[16, 7, 8 * 512, 35]])
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
            # res_cpu.backward(grad_cpu)
            # print("--a={}".format(a))
            with torch.autograd.profiler_legacy.profile(profiling, use_xpu=True) as prof:
                a_xpu = a.clone().to("xpu").to(dtype)
                a_xpu.requires_grad_()
                grad_xpu = grad.clone().to("xpu").to(dtype)
                res_xpu = F.softmax(a_xpu, dim)
                res_xpu.backward(grad_xpu)
            if profiling:
                print(prof.table(sort_by="id", row_limit=100000)) 

            diff1 = torch.max(torch.abs(res_xpu.cpu() - res_cpu)[0, 0, :])
            diff2 = torch.max(torch.abs(res_xpu.cpu() - res_cpu)[0, 1, :])
            diff = torch.max(torch.abs(res_xpu.cpu() - res_cpu))
            # diff_grad = torch.max(torch.abs(a_xpu.grad.cpu() - a_cpu.grad))
            print("---diff={}".format(diff))
            # print("---diff_grad={}".format(diff_grad))

