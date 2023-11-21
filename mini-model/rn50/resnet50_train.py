import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import intel_extension_for_pytorch
import time
import os

need_profile = False 
channels_last = True
bs=256
device = torch.device("xpu")
datatype = torch.bfloat16

train_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=bs, shuffle=False,
    num_workers=4, pin_memory=True, pin_memory_device="xpu", sampler=None)

# create model
model = models.resnet50(pretrained=True)  # create model
model.train()

# model to device
model = model.xpu() 
criterion = torch.nn.CrossEntropyLoss().xpu()

#create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.001, weight_decay=0.001) 
# enable ipex.optimize for performance acceleration
if not channels_last:
    intel_extension_for_pytorch.disable_auto_channels_last()
model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, dtype=datatype)

warmup_iter = 5
iteration = 15
total_time = 0
import contextlib
def profiler_setup(need_profile):
    print("need_profile={}".format(need_profile))
    if need_profile:
        return torch.profiler.profile(
            activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU],
            schedule=torch.profiler.schedule(
            skip_first=iteration-5,
            wait=1,
            warmup=3,
            active=1),
            record_shapes=True,
            profile_memory=False,  
            with_stack=True
        )
    else:
        return contextlib.nullcontext()


# step3. do training
with profiler_setup(need_profile) as prof:
    for step, (images, target) in enumerate(train_loader):
        start_time = time.time()
    
        # H2D
        images = images.to(device=device)
        target = target.to(device=device)
    
        # forward
        with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
            outputs = model(images)
            loss = criterion(outputs, target)
    
        # backward and SGD step
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        optimizer.step()
    
        # D2H
        loss = loss.cpu()
        output = outputs.cpu()
    
        # sync
        torch.xpu.synchronize()
        end_time = time.time()
        if step == iteration:
            break
        if need_profile:
            prof.step()

        # compute performance
        if step >= warmup_iter:
            total_time += end_time - start_time
            latency = total_time / (step - warmup_iter + 1)
            throughput = bs / latency
            print("---latency={} s".format(latency))
            print("---throughput={} fps".format(throughput))

    if need_profile:
        print(str(prof.key_averages().table(sort_by="self_xpu_time_total")))
        torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'train_profiling.pt')
        prof.export_chrome_trace('./train_profile_trace.json')
    

