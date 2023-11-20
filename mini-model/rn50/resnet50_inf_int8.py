import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import intel_extension_for_pytorch
import time

bs=1
channels_last = True
device = torch.device("xpu")
datatype = torch.float16
datatype = torch.int8

val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=bs, shuffle=False,
    num_workers=4, pin_memory=True, pin_memory_device="xpu", sampler=None)

# step1. prepare model
model = models.resnet50(pretrained=True)
model.eval()
model = model.xpu()
if channels_last:
    model = model.to(memory_format=torch.channels_last)

sample_input = torch.randn([1, 3, 224, 224]).to(device)
if datatype == torch.int8:
    # step2. enable xpu.optimize
    model = torch.xpu.optimize(model=model, dtype=torch.float)

    # step3. enable jit + calibration
    from torch.quantization.quantize_jit import (
        convert_jit,
        prepare_jit,
    )
    modelJit = torch.jit.trace(model, sample_input)
    with torch.no_grad():
        qconfig = torch.quantization.QConfig(
            activation=torch.quantization.observer.MinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric,
                reduce_range=False,
                dtype=torch.quint8),
                weight=torch.quantization.default_weight_observer)
        # insert observer
        modelJit = prepare_jit(modelJit, {'': qconfig}, True)
        for i, (input, target) in enumerate(val_loader):
            calib = input.to("xpu")
            modelJit(calib)
            if i == 3:
                break
        # fusion
        modelJit = convert_jit(modelJit, True)
else:
    # step2. enable xpu.optimize
    model = torch.xpu.optimize(model=model, dtype=datatype)

    # step3. enable amp + jit
    with torch.no_grad():
        with torch.xpu.amp.autocast(enabled=True, dtype=datatype, cache_enabled=False):
            modelJit = torch.jit.trace(model, sample_input)

    # save and load
    torch.jit.save(modelJit, "scriptmodule.pt")
    modelJit = torch.jit.load("scriptmodule.pt")

# step4. do inference e2e
with torch.no_grad():
    for step, (images, target) in enumerate(val_loader):
        # print("step:{}".format(step))
        # print(modelJit.graph_for(sample_input))
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        start_time = time.time()
        # H2D
        images = images.to(device=device)
        # forward
        outputs = modelJit(images)
        # sync
        torch.xpu.synchronize()
        end_time = time.time()

        latency = end_time - start_time
        throughput = bs / latency
        print("---latency={} ms".format(latency))
        print("---throughput={} fps".format(throughput))
        if step + 1 >= 10:
            break

