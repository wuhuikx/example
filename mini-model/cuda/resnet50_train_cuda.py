import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.models as models

import torch.profiler

bool channels_last = True
model = models.resnet50(pretrained=True)
model.cuda()
cudnn.benchmark = True

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0")
model.train()
if channels_last:
    model = model.to(memory_format=torch.channels_last)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./result', worker_name='worker0'),
    record_shapes=True,
    profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    with_stack=True
) as p:
    for step, data in enumerate(trainloader, 0):
        print("step:{}".format(step))
        inputs, labels = data[0].to(device=device), data[1].to(device=device)
        if channels_last:
            input = input.to(memory_format=torch.channels_last)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step + 1 >= 4:
            break
        p.step()
print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=200000))
