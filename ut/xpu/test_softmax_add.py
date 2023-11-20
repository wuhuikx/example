import torch
import torch.nn.functional as F
import numpy as np
import intel_extension_for_pytorch


dim = 2
dtypes = [torch.bfloat16, torch.float]
dtypes = [torch.float]
#dtypes = [torch.bfloat16]
profiling = False 

class AddSoftmax(torch.nn.Module):
    def __init__(self, dim):
        super(AddSoftmax, self).__init__()
        self.softmax = torch.nn.Softmax(dim)

    def forward(self, x):
        x1 = x.clone() * 5
        y = x + x1
        y = self.softmax(y)
        return y

for index, dtype in enumerate(dtypes):
    input_shape = np.array([[1, 2, 3], [16, 5, 17], [256, 32, 1023], [256, 11, 255], [16*16, 512, 512], [16*16, 512, 513], [1, 8192, 8192], [1, 8192, 8732], [1, 8192, 30522], [128, 81, 8732], [1024, 128, 1000]])
    #input_shape = np.array([[256,512,256], [1,512,256], [256,512,512], [1,512,512], [256,512,1024], [1,512,1024], [256,512,2048], [1,512,2048], [1,128,1000], [16,128,1000], [1, 8732, 8732], [1, 8000, 8000], [1,8732,30522], [1, 8732, 30000], [1,1,30522]])
    #input_shape = np.array([[256,512,512], [512, 384, 384], [1, 8192, 30522], [128, 81, 8732]])
    input_shape = np.array([[512, 3000, 21], [512,3000,91], [256, 512, 512], [512, 384, 384], [1, 64, 30522], [1, 64, 8192], [1, 8192, 64], [28, 4096, 8], [2048, 64, 3], [1024, 128, 1000]])
    #input_shape = np.array([[512,3000,21], [512, 3000, 91]])
    # input_shape = np.array([[512, 3000, 21], [512,3000,91]])
    # input_shape = np.array([[256, 512, 512]])
    # input_shape = np.array([[512, 384, 384]])
    #input_shape = np.array([[1, 64, 8192]])
    #input_shape = np.array([[1, 8192, 64]])
    #input_shape = np.array([[512,3000,21]])
    #input_shape = np.array([[28, 4096, 8]])
    #input_shape = np.array([[28, 4095, 8]])
    #input_shape = np.array([[16, 5, 17]])
    #input_shape = np.array([[16, 7, 8 * 512, 35]])
    iteration = 1
    # accuracy
    for i in range(iteration):
        for j in range(input_shape.shape[0]):
            size = input_shape[j]
            print("acc, dtype={}, size={}, dim={}".format(dtype, size, dim))
            a = torch.randn(tuple(size))
            a_cpu = a.clone()
            a_xpu = a.clone().to("xpu").to(dtype)
            model_cpu = AddSoftmax(1)
            model_cpu.eval()
            model_xpu = AddSoftmax(1)
            model_xpu.eval()
            model_xpu.to("xpu")
            modelJit = torch.jit.trace(model_xpu, a_xpu)
            print(modelJit.graph_for(a_xpu))

            # print("--a={}".format(a))
            with torch.autograd.profiler_legacy.profile(profiling, use_xpu=True) as prof:
                res_cpu = model_cpu(a_cpu)
                #res_xpu = model_xpu(a_xpu)
                res_xpu = modelJit(a_xpu)
            if profiling:
                print(prof.table(sort_by="id", row_limit=100000)) 

            diff = torch.max(torch.abs(res_xpu.cpu() - res_cpu))
            print("---diff={}".format(diff))

