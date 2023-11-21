import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import intel_extension_for_pytorch
import time
import os

profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]

#channels_last = True
channels_last = False
bs=32
device = torch.device("xpu")
#datatype = torch.bfloat16
datatype = torch.float

train_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=bs, shuffle=False,
    num_workers=4, pin_memory=True, pin_memory_device="xpu", sampler=None)

# step1. prepare model
model = models.resnet50(pretrained=True)
model.train()
model = model.xpu()
criterion = torch.nn.CrossEntropyLoss().xpu()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.001, weight_decay=0.001)

# step2. enable xpu.optimize
if not channels_last:
    intel_extension_for_pytorch.disable_auto_channels_last()
model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, dtype=datatype)

warmup_iter = 5
total_time = 0
# step3. do training
with torch.autograd.profiler.emit_itt():
    for step, (images, target) in enumerate(train_loader):
        start_time = time.time()
        with torch.autograd.profiler_legacy.profile(profiling, use_xpu=True) as prof:
            torch.profiler.itt.range_push('step_{}'.format(step))
            import sys
            sys.stdout.flush()
            # H2D
            images = images.to(device=device)
            target = target.to(device=device)
            # forward
            with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
                outputs = model(images)
                loss = criterion(outputs, target)
            # backward and SGD step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # D2H
            loss = loss.cpu()
            output = outputs.cpu()
            # sync
            torch.xpu.synchronize()
            torch.profiler.itt.range_pop()
        end_time = time.time()

        if profiling:
            torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profile.pt')
            torch.save(prof.table(sort_by="id", row_limit=-1), 'profile_detailed.pt')
            torch.save(prof.key_averages(group_by_input_shape=True).table(sort_by="self_xpu_time_total", row_limit=200000), 'profile_shape.pt')
            prof.export_chrome_trace('./profile_trace.json')

        # compute performance
        if step >= warmup_iter:
            total_time += end_time - start_time
            latency = total_time / (step - warmup_iter + 1)
            throughput = bs / latency
            print("---latency={} s".format(latency))
            print("---throughput={} fps".format(throughput))
        if step + 1 >= 500:
            break

