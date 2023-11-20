import torch
import intel_extension_for_pytorch
import os


format=torch.contiguous_format
format=torch.channels_last
input_shapes = [
        [257, 1024], 
        [1, 1024], 
        [2, 4096, 320], 
        [2, 1024, 640], 
        [2, 256, 1280], 
        [2, 64, 1280], 
        [8192, 1024], 
        [196, 512], 
        [49, 1024], 
        [49, 2048], 
        [784, 256], 
        [784, 512], 
        [3136, 128], 
        [16384, 1024], 
        [2432, 1024], 
        [128, 4096],
        [4, 4096], 
        [24576, 1024], 
        [16384, 768], 
        [16384, 3072],
        [257, 1023], 
        [257, 1025], 
        [257, 7], 
        [1024, 512],
        [1024, 255],
        [32, 2048*16*15 +1],
        [32, 2048*16*16 +1],
        [1024, 384, 385],
        [1024, 384, 385],
        [20, 5, 10, 10]]
norm_shapes = [
        [1024], 
        [1024], 
        [320], 
        [640], 
        [1280], 
        [1280], 
        [1024], 
        [512], 
        [1024], 
        [2048], 
        [256], 
        [512], 
        [128], 
        [1024], 
        [1024], 
        [4096],
        [4096],
        [1024], 
        [768],
        [3072],
        [1023], 
        [1025], 
        [7], 
        [512],
        [255],
        [2048*16*15 +1],
        [2048*16*16 +1],
        [384, 385],
        [385],
        [5, 10, 10]]
input_shapes = [[20, 5, 10, 10]]
norm_shapes = [[5, 10, 10]]
#input_shapes = [[2, 4096, 320], [1, 32, 2048*8]]
#norm_shapes = [[320], [2048*8]]

#input_shapes = [[1, 257, 1023], [1, 257, 1025], [1, 257, 7], [1, 1024, 255]]
#norm_shapes = [[1023], [1025], [7], [255]]
#input_shapes = [[1, 128, 2048]]
#norm_shapes = [[2048]]
#input_shapes = [[1, 128, 2048*4]]
#norm_shapes = [[2048*4]]
#input_shapes = [[1, 128, 2048*6]]
#norm_shapes = [[2048*6]]
#input_shapes = [[1, 128, 2048*8]]
#norm_shapes = [[2048*8]]
#input_shapes = [[1, 128, 2048*16+1]]
#norm_shapes = [[2048*16+1]]
#input_shapes = [[1, 32, 2048*16*16 +1]]
#norm_shapes = [[2048*16*16+1]]
profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
dtype = torch.float
# dtype = torch.float16
#input = torch.ones(shape)
with torch.autograd.profiler_legacy.profile(profiling, use_xpu=True, record_shapes=True) as prof:
    for idx, input_shape in enumerate(input_shapes):
        norm_shape = norm_shapes[idx]
        print("=========input_shape={}".format(input_shape))
        print("=========norm_group={}".format(norm_shape))
        input = torch.ones(input_shape)
        input = torch.randn(input_shape)
        grad = torch.randn(input_shape)
        if (input.dim() == 4):
            input = input.to(memory_format=format)
            grad = grad.to(memory_format=format)
        
        # mean = torch.mean(input, dim=2)
        # print("---mean={}".format(mean))

        input_cpu = input.clone()
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone() 
        m = torch.nn.LayerNorm(norm_shape)
        output_cpu = m(input_cpu)
        output_cpu.backward(grad_cpu)
        grad_wei = m.weight.grad.clone()
        
        # input_cpu = input.to(dtype)
        # model_cpu = m.to(dtype)
        # output_cpu = model_cpu(input_cpu)

        #shape_view = [int(shape[0]*group), int(shape[1]/group * shape[2] * shape[3])]
        #input_view = input.view(shape_view)
        #print("--mean={}".format(torch.mean(input_view[0, :])))
        #print("--mean={}".format(torch.mean(input_view[2, :])))
        for i in range(1):
            m.zero_grad()
            input_xpu = input.clone().to("xpu").to(dtype)
            input_xpu.requires_grad_(True)
            grad_xpu = grad.clone().to("xpu")
            model_xpu = m.to("xpu").to(dtype)
            model_xpu.zero_grad()
            output_xpu = model_xpu(input_xpu)
            output_xpu.backward(grad_xpu)
            grad_wei_xpu = model_xpu.weight.grad.clone()

        #print("cpu={}".format(output_cpu.float()))
        #print("xpu={}".format(output_xpu.float().cpu()))
        diff = torch.max(torch.abs(output_cpu.float().contiguous() - output_xpu.float().cpu().contiguous()))
        print("----diff={}".format(diff))
        diff_grad = torch.max(torch.abs(input_cpu.grad.float().contiguous() - input_xpu.grad.float().cpu().contiguous()))
        print("----diff_grad={}".format(diff_grad))
        
        #print("----grad_wei={}".format(grad_wei))
        print("----grad_wei_xpu={}".format(grad_wei_xpu))
        diff_wei_grad = torch.max(torch.abs(grad_wei.cpu().contiguous() - grad_wei_xpu.cpu().contiguous()))
        print("----diff_wei_grad={}".format(diff_wei_grad))
        #  diff_dtype = torch.max(torch.abs(output_cpu.float() - output.float()))
        # print("----diff_dtype={}".format(diff_dtype))
if profiling:
    torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profile.pt')
    torch.save(prof.key_averages(group_by_input_shape=True).table(sort_by="self_xpu_time_total", row_limit=200000), 'profile_shape.pt')
    torch.save(prof.table(sort_by="id", row_limit=-1), 'profile_detailed.pt')
    prof.export_chrome_trace('./profile_trace.json')

    
