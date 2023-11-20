import torch
import intel_extension_for_pytorch

p=0.1
input=torch.randn([16, 512, 1024])
input = input.fill_(1-p)
output=torch.nn.functional.dropout(input, p=0.1)
print("output={}".format(output))

input_xpu=input.to("xpu").bfloat16()
#input_xpu=input.to("xpu")
output=torch.nn.functional.dropout(input_xpu, p=0.1)
output_cpu=output.cpu()
print("output_cpu={}".format(output_cpu))

print(torch.abs(output_cpu.data.mean()-(1-p)))
