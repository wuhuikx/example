import torch
import intel_extension_for_pytorch
import torch.nn as nn
import torch.nn.functional as F

torch._C._jit_set_profiling_mode(False)                                         
torch._C._jit_set_profiling_executor(False)   

device='xpu'
#device='cpu'
#dtype=torch.float32
dtype=torch.bfloat16
torch.random.manual_seed(123)
bs = 512

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1024):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                #nn.BatchNorm2d(oup),
                #nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                #nn.BatchNorm2d(inp),
                #nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                #nn.BatchNorm2d(oup),
                #nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            #conv_dw(64, 128, 2),
            #conv_dw(128, 128, 1),
            #conv_dw(128, 256, 2),
            #conv_dw(256, 256, 1),
            #conv_dw(256, 512, 2),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 1024, 2),
            #conv_dw(1024, 1024, 1),
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        #x = F.avg_pool2d(x, 7)
        #x = x.view(-1, 1024)
        #x = self.fc(x)
        return x


#print('\n***** create loss function')
#criterion = nn.CrossEntropyLoss().to(device)
print('\n***** create model')
model_dpcpp = MobileNetV1()
model_dpcpp = model_dpcpp.to(device=device)
model_dpcpp.eval()
#optimizer = torch.xpu.optim.SGDMasterWeight(model_dpcpp.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
#model_dpcpp = model_dpcpp.bfloat16()

input = torch.randn(512, 3, 300, 300)
input_dpcpp = input.to("xpu")

#modelJit = torch.jit.trace(model_dpcpp, input_dpcpp)
modelJit = model_dpcpp
NUM_ITER=2
for iteration in range(NUM_ITER):
    print("--iteration={}".format(iteration))
    with torch.inference_mode():
        output_dpcpp = modelJit(input_dpcpp)
    output_xpu = output_dpcpp.to("cpu")
    #loss = criterion(output_dpcpp, target_dpcpp)
    #optimizer.zero_grad(set_to_none=True)
    #loss.backward()
    #optimizer.step()
    #torch.xpu.synchronize()
    #print("Iter[", iteration, "] loss = ", loss.cpu().item())

