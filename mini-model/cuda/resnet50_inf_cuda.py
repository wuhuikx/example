import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.models as models

import torch.profiler
import time
# Enabling the TorchScript Runtime Backend NVFuser
#torch._C._jit_set_nvfuser_enabled(True)
#torch._C._jit_set_texpr_fuser_enabled(False)
#torch._C._jit_override_can_fuse_on_cpu(False)
#torch._C._jit_override_can_fuse_on_gpu(False)
#torch._C._jit_set_autocast_mode(True)
#torch._C._jit_set_bailout_depth(20)


model = models.resnet50(pretrained=True)
model.cuda()
cudnn.benchmark = True
bs=32
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                          shuffle=True, num_workers=4)

device = torch.device("cuda:0")
model.eval()
model.half()

# step1. enable jit fusion
sample_input = torch.randn([1, 3, 224, 224]).cuda().half()
with torch.no_grad():
    # model = torch.jit.script(model)
    model = torch.jit.trace(model, sample_input)
    model = torch.jit.optimize_for_inference(model) # conv bn folding, dropout removal

# step2. do inference e2e
with torch.no_grad():
    for step, data in enumerate(trainloader, 0):
        print("step:{}".format(step))
        print("bs={}".format(data[0].shape))

        start_time = time.time()
        # H2D
        inputs, labels = data[0].to(device=device), data[1].to(device=device)
        # datatype cast
        inputs = inputs.half()
        # forward
        outputs = model(inputs)
        # sync
        torch.cuda.synchronize()
        end_time = time.time()

        latency = end_time - start_time
        throughput = bs / latency
        print("---latency={} ms".format(latency))
        print("---throughput={} fps".format(throughput))
        if step + 1 >= 20:
            break

# step3. do inference profile
with torch.no_grad():
    for step, data in enumerate(trainloader, 0):
        print("step:{}".format(step))
        # step3.1. add profling
        with torch.profiler.profile(
            activities=[
               torch.profiler.ProfilerActivity.CPU,
               torch.profiler.ProfilerActivity.CUDA],
            #schedule=torch.profiler.schedule(
            #    wait=1,
            #    warmup=1,
            #    active=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./result', worker_name='worker0'),
            record_shapes=True,
            profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
            with_stack=True
        ) as p:

            # step3.2. H2D
            inputs, labels = data[0].to(device=device), data[1].to(device=device)
            # step3.3. datatype cast
            inputs = inputs.half()
            # step3.4. forward
            outputs = model(inputs)
            # step3.5. sync
            torch.cuda.synchronize()

            if step + 1 >= 4:
                break
            p.step()

        # step3.6. print profiling result
        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=200000))
#print(model.graph_for(sample_input))

