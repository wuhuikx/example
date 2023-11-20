import torch
import intel_extension_for_pytorch

from torch.nn.modules.utils import _pair

zero_point = 0
dtype_inputs = torch.quint8
dtype_filters = torch.qint8

scale_in = 0.4
scale_weight = 0.5
scale_out = 4.0
scale_out_2 = 8.0

inputs = torch.randn(1, 2, 5, 5)
filters = torch.randn(4, 2, 3, 3)
bias = torch.randn(4)

with torch.autograd.profiler_legacy.profile(True, use_xpu=True, with_calling_stack=True) as prof:
    q_inputs = torch.quantize_per_tensor(inputs, scale_in, zero_point, dtype_inputs)
    q_filters = torch.quantize_per_tensor(filters, scale_weight, zero_point, dtype_filters)

    packed_params = torch.ops.quantized.conv2d_prepack(q_filters, bias, _pair(1), _pair(0), _pair(1), 1)
    #output_int8 = torch.ops.quantized.conv2d_relu(q_inputs, packed_params, _pair(1), _pair(0), _pair(1), 1, scale_out, zero_point)
    output_int8_ref = torch.ops.quantized.conv2d(q_inputs, packed_params, _pair(1), _pair(0), _pair(1), 1, scale_out, zero_point)
print(prof.table(sort_by="id", max_depth=10, row_limit=10000000))

print("output_int8_ref={}".format(output_int8_ref))
output = torch.nn.functional.leaky_relu(output_int8_ref, 0.5)
print("output={}".format(output))

'''
inputs_gpu=inputs.to("xpu")
filters_gpu = filters.to("xpu")
bias_gpu = bias.to("xpu")

q_inputs_gpu = torch.quantize_per_tensor(inputs_gpu, scale_in, zero_point, dtype_inputs)
q_filters_gpu = torch.quantize_per_tensor(filters_gpu, scale_weight, 128, dtype_filters)

packed_params_gpu = torch.ops.quantized.conv2d_prepack(q_filters_gpu, bias_gpu, _pair(1), _pair(0), _pair(1), 1)
#output_int8 = torch.ops.quantized.conv2d_relu(q_inputs_gpu, packed_params, _pair(1), _pair(0), _pair(1), 1, scale_out, zero_point)

#output_int8_gpu = torch.ops.quantized.conv2d_relu(q_inputs_gpu, packed_params_gpu, scale_out, zero_point)
#output_int8_gpu = intel_extension_for_pytorch._C.qconv2d_leakyrelu(q_inputs_gpu, packed_params_gpu, scale_out, zero_point, 0.5)

#output_gpu = torch.nn.functional.leaky_relu(output_int8_gpu, 0.5)
print("output_input8_gpu={}".format(output_int8_gpu))
#print("output_gpu={}".format(output_gpu))
'''
