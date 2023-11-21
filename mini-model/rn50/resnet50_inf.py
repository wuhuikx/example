import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import intel_extension_for_pytorch
import time

bs=1024
device = torch.device("xpu")
datatype = torch.float16
jit = True
dynamo = False 
need_profile = False 

val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=bs, shuffle=False,
    num_workers=4, pin_memory=True, pin_memory_device="xpu", sampler=None)

# prepare model and data
model = models.resnet50(pretrained=True)
model.eval()
model = model.xpu()

# enable xpu.optimize
model = torch.xpu.optimize(model=model, dtype=datatype)

# enable graph mode 
if jit:
    sample_input = torch.randn([bs, 3, 224, 224]).to(device)
    with torch.no_grad():
        with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
            model_graph = torch.jit.trace(model, sample_input)
            model_graph = torch.jit.freeze(model_graph)
elif dynamo:
    model_graph = torch.compile(model, backend="inductor", options={"freezing": True})
else:
    model_graph = model


total_time = 0
warmup_iter = 5
iteration = 15
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

# step4. do inference e2e
with profiler_setup(need_profile) as prof:
    with torch.no_grad():
        for step, (images, target) in enumerate(val_loader):
            start_time = time.time()
            # H2D
            images = images.to(device=device)
            # forward
            with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
                outputs = model_graph(images)
            # D2H
            outputs = outputs.cpu()
            # sync
            torch.xpu.synchronize()
            end_time = time.time()
            if step == iteration:
                break
            if need_profile:
                prof.step()

            if step >= warmup_iter:
                total_time += end_time - start_time
                latency = total_time / (step - warmup_iter + 1)
                throughput = bs / latency
                print("---latency={} s".format(latency))
                print("---throughput={} fps".format(throughput))

    if need_profile:
        print(str(prof.key_averages().table(sort_by="self_xpu_time_total")))
        torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'inf_profiling.pt')
        prof.export_chrome_trace('./inf_profile_trace.json')
    
