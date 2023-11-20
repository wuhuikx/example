import torch
import intel_extension_for_pytorch

input=torch.tensor(
        [[[ 0.5617,  0.6389, -0.3111,  0.7591],
         [-0.5706, -0.9129,  0.6084,  0.3192],
         [-0.0511,  0.8164,  0.2019,  0.0281]],
        [[-0.8874,  0.0371,  0.3604, -0.9179],
         [ 0.4932,  0.5771, -0.3977,  0.4833],
         [-0.2288,  0.4286, -0.9623,  0.8964]]]).double()
#out=torch.nn.functional.interpolate(input, scale_factor=3, mode='linear', align_corners=True)
#print("out_size={}".format(out.size()))
#grad = torch.randn([2, 3, 12]).double()
#grad_xpu = grad.clone().xpu()
input_cpu = input.clone()
input_xpu = input.clone().xpu()
input_cpu.requires_grad_(True)
input_xpu.requires_grad_(True)


#out_cpu=torch.nn.functional.interpolate(input_cpu, scale_factor=3, mode='linear', align_corners=True)
#out_cpu.backward(torch.ones(out_cpu.size()).double())
#print("out_cpu={}".format(out_cpu))

out_xpu=torch.nn.functional.interpolate(input_xpu, scale_factor=3, mode="linear", align_corners=True)
out_xpu.backward(torch.ones(out_xpu.size()).double().xpu())
print("out_xpu={}".format(out_xpu))

#diff1 = torch.max(torch.abs(out_cpu - out_xpu.cpu()))
#diff2 = torch.max(torch.abs(input.grad - input_xpu.grad.cpu()))
#print(diff1)
#print(diff2)
