import torch
from torch.testing._internal.common_utils import TestCase
from torch.nn.modules.utils import _pair
import time
import intel_extension_for_pytorch

from torch.jit._recursive import wrap_cpp_module
from torch.quantization.quantize_jit import (
    convert_jit,
    prepare_jit,
)
from torch.quantization import default_qconfig





#torch._C._jit_set_profiling_mode(False)
#torch._C._jit_set_profiling_executor(False)
device = "xpu"
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv = torch.nn.Conv2d(3, 128, kernel_size=7, stride=1, padding=0, bias=True)
        self.leaky_relu = torch.nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        return x

model = M()

inputs = torch.randn(2, 3, 16, 16)
filters = model.conv.weight.detach().clone()
bias = model.conv.bias.detach().clone()

inputs_gpu = inputs.to("xpu")
filters_gpu = filters.to("xpu")
bias_gpu = bias.to("xpu")

conv_fp32 = torch.nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0, bias=True)
Xelu = torch.nn.LeakyReLU(0.1, inplace=False)
conv_fp32 = conv_fp32.to("xpu")
conv_fp32.weight.data = filters_gpu.data
conv_fp32.bias.data = bias_gpu.data
output_fp32 = conv_fp32(inputs_gpu)

dtype_inputs = torch.qint8
dtype_filters = torch.qint8
scale_in = 0.025239113718271255
#scale_in = torch.max(torch.abs(inputs_gpu)) / 127
scale_weight = torch.max(torch.abs(filters_gpu)) / 127
#scale_out = torch.max(torch.abs(output_fp32)) / 127
scale_out = 0.019031498581171036
zero_point = 0

print("scale_in={}".format(scale_in))
print("scale_weight={}".format(scale_weight))
print("scale_out={}".format(scale_out))

q_inputs_gpu = torch.quantize_per_tensor(inputs_gpu, scale_in, zero_point, dtype_inputs)
q_filters_gpu = torch.quantize_per_tensor(filters_gpu, scale_weight, zero_point, dtype_filters)
packed_params_gpu = torch.ops.quantized.conv2d_prepack(q_filters_gpu, bias_gpu, _pair(1), _pair(0), _pair(1), 1)
output_gpu_int8 = torch.ops.quantized.conv2d(q_inputs_gpu, packed_params_gpu, scale_out, zero_point)
output_gpu_int8 = Xelu(output_gpu_int8)
gpu_result = torch.dequantize(output_gpu_int8)
'''

inputs_xpu = input_fp32.clone().to(device)
filters_xpu = filters.clone().to(device)
bias_xpu = bias.clone().to(device)

conv_fp32 = torch.nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=True)
conv_fp32.weight.data = filters.data
conv_fp32.bias.data = bias.data
output_fp32 = conv_fp32(input_fp32)

dtype_inputs = torch.qint8
dtype_filters = torch.qint8
scale_in = 128.0 / torch.max(torch.abs(input_fp32))
scale_weight = 128.0 / torch.max(torch.abs(filters))
scale_out = 128.0 / torch.max(torch.abs(output_fp32))
zero_point = 128

q_inputs = torch.quantize_per_tensor(inputs_xpu, scale_in, zero_point, dtype_inputs)
q_filters = torch.quantize_per_tensor(filters_xpu, scale_weight, zero_point, dtype_filters)
packed_params = torch.ops.quantized.conv2d_prepack(q_filters, bias_xpu, _pair(1), _pair(0), _pair(1), 1)
output_xpu = torch.ops.quantized.conv2d_relu(q_inputs, packed_params, _pair(1), _pair(0), _pair(1), 1, scale_out, zero_point)
'''


print("start jit ...")
modelJit = torch.jit.trace(model, inputs)
modelJit.eval()
print("finish jit ...")

modelJit = modelJit.to(device)

print("start calibration ...")
# calibration
# default with per_tensor quantization
with torch.inference_mode():
    qconfig_u8 = torch.quantization.QConfig(
        activation=torch.quantization.observer.MinMaxObserver.with_args(
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False,
            dtype=torch.qint8
        ),
        weight=torch.quantization.default_weight_observer
    )

    modelJit = prepare_jit(modelJit, {'': qconfig_u8}, True)
    modelJit = modelJit.to(device)

    # do calibration
    for i in range(1):
        calib_input = inputs_gpu
        modelJit(calib_input)
        modelJit = convert_jit(modelJit, True)
        print(modelJit.graph_for(inputs_gpu))

    print("----model-----")
    for name, module in modelJit.named_children():
        print(name)
        print(module)
    # inference
    print("start inference ...")
    with torch.inference_mode():
        for i in range(5):
            start = time.time()

            output = modelJit(inputs_gpu)
            if device == "xpu":
                torch.xpu.synchronize()

            end = time.time()
            print("iter.{} ... {time:.3f}ms".format(i, time=(end - start) * 1000))
print("--gpu_result={}".format(output.cpu()))
print("--gpu_result={}".format(gpu_result.cpu()))
diff = torch.max(torch.abs((output.cpu() - gpu_result.cpu())))
print("diff={}".format(diff))
