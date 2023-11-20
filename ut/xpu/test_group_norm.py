import torch
import intel_extension_for_pytorch
import os


#shapes = [[1, 128, 512, 512]]
#groups = [256, 128, 64, 32, 16, 4, 2]
shapes = [[2, 2560, 32, 32], 
            [2, 2560, 16, 16],
            [2, 2560, 8, 8],
            [2, 1920, 32, 32],
            [2, 1920, 16, 16],
            [2, 1920, 8, 8],
            [2, 1280, 32, 32],
            [2, 1280, 16, 16],
            [2, 1280, 8, 8],
            [2, 960, 64, 64],
            [2, 960, 32, 32],
            [2, 960, 16, 16],
            [2, 640, 64, 64],
            [2, 640, 32, 32],
            [2, 640, 16, 16],
            [2, 320, 64, 64],
            [1, 512, 128, 128],
            [1, 512, 64, 64],
            [1, 256, 256, 256],
            [1, 128, 512, 512],
            [1, 256, 513, 513],
            [1, 128, 512, 512],
            [1, 256, 55,55]]
#shapes = [[1, 256, 513, 513], [1, 256, 1021, 1023], [1, 128, 512, 512], [1, 256, 55, 55], [1, 128, 7, 7]]
#shapes=[[1, 256, 1021, 1023]]
#groups = [128]
groups = [128, 32]

format=torch.contiguous_format
format=torch.channels_last
#shapes = [[2, 1280, 8, 8]]
#groups = [1280]
#shapes = [[1, 256, 513, 513]]
#groups = [32]
#groups = [128]

profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
dtype = torch.float
#dtype = torch.float16
#input = torch.ones(shape)
for shape in shapes:
    for group in groups:
        group = min(group, shape[1])
        if (shape[1]%group):
          continue 
        print("=========shape={}".format(shape))
        print("=========group={}".format(group))
        input = torch.ones(shape)
        input = torch.randn(shape)
        grad = torch.randn(shape)
        input = input.to(memory_format=format)
        grad = grad.to(memory_format=format)
        
        input_cpu = input.clone()
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone() 
        m = torch.nn.GroupNorm(group, shape[1])
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
        with torch.autograd.profiler_legacy.profile(profiling, use_xpu=True) as prof:
            for i in range(1):
                input_xpu = input.clone().to("xpu").to(dtype)
                input_xpu.requires_grad_(True)
                grad_xpu = grad.clone().to("xpu")
                model_xpu = m.to("xpu").to(dtype)
                model_xpu.zero_grad()
                output_xpu = model_xpu(input_xpu)
                output_xpu.backward(grad_xpu)
                grad_wei_xpu = model_xpu.weight.grad.clone()
        if profiling:
            torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profile.pt')
            torch.save(prof.table(sort_by="id", row_limit=-1), 'profile_detailed.pt')
            prof.export_chrome_trace('./profile_trace.json')

        diff = torch.max(torch.abs(output_cpu.float() - output_xpu.float().cpu()))
        print("----diff={}".format(diff))
        diff_grad = torch.max(torch.abs(input_cpu.grad.float() - input_xpu.grad.float().cpu()))
        print("----diff_grad={}".format(diff_grad))
        diff_wei_grad = torch.max(torch.abs(grad_wei.cpu().contiguous() - grad_wei_xpu.cpu().contiguous()))
        print("----diff_wei_grad={}".format(diff_wei_grad))

