import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import intel_extension_for_pytorch
import time

bs=1024
device = torch.device("xpu")
datatype = torch.float16

val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=bs, shuffle=False,
    num_workers=4, pin_memory=True, pin_memory_device="xpu", sampler=None)

# step1. prepare model and data
model = models.resnet50(pretrained=True)
model.eval()
model = model.xpu()
sample_input = torch.randn([bs, 3, 224, 224]).to(device)

# step2. enable xpu.optimize
model = torch.xpu.optimize(model=model, dtype=datatype)

# step3. enable jit + amp
with torch.no_grad():
    with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
        modelJit = torch.jit.trace(model, sample_input)
        modelJit = torch.jit.freeze(modelJit)

warmup_iter = 5
total_time = 0
# step4. do inference e2e
with torch.no_grad():
    for step, (images, target) in enumerate(val_loader):
        start_time = time.time()
        # H2D
        images = images.to(device=device)
        # forward
        outputs = modelJit(images)
        # D2H
        outputs = outputs.cpu()
        # sync
        torch.xpu.synchronize()
        end_time = time.time()

        if step >= warmup_iter:
            total_time += end_time - start_time
            latency = total_time / (step - warmup_iter + 1)
            throughput = bs / latency
            print("---latency={} s".format(latency))
            print("---throughput={} fps".format(throughput))
        if step + 1 >= 500:
            break

