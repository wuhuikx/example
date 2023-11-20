import torch
import intel_extension_for_pytorch
import torch.nn as nn


dpcpp_device = "xpu"
cpu_device = "cpu"
class MatmulRelu(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulRelu, self).__init__()

    def forward(self, m1, m2):
        return  torch.relu(torch.matmul(m1, m2))

class MatmulSum(torch.nn.Module):
    def __init__(self):
        super(MatmulSum, self).__init__()

    def forward(self, m1, m2, a): 
        y = torch.matmul(m1, m2) 
        y += a
        return y


def matmul_fusion(model, m1, m2):
    raw = model(m1.clone(), m2.clone())
    m1_dpcpp = m1.to(dpcpp_device)
    m2_dpcpp = m2.to(dpcpp_device)
    model = model.to(dpcpp_device)
    modelJit = torch.jit.script(model)
    for i in range(2):
        modelJit(m1_dpcpp, m2_dpcpp)
    with torch.no_grad():
        print(modelJit.graph_for(m1_dpcpp, m2_dpcpp))
        real = modelJit(m1_dpcpp, m2_dpcpp)
    print("--cpu_shape={}".format(raw.shape))
    print("--xpu_shape={}".format(real.cpu().shape))
    diff = torch.max(torch.abs(raw.to(cpu_device) - real.to(cpu_device)))
    print("---diff={}".format(diff))
    #self.assertEqual(raw, real.to(cpu_device))
    #self.assertEqual(raw_t, t_real.to(cpu_device))



def test_matmul_sum_fusion(shapes, dtype=torch.float):
        # case1
    for shape in shapes:
        m1 = torch.randn(shape[0], device=cpu_device)
        m2 = torch.randn(shape[1], device=cpu_device)
        acc = torch.randn(shape[2], device=cpu_device)
        print("m1:", m1.shape)
        print("m2:", m2.shape)
        print("acc:", acc.shape)
        m1_dpcpp_orig = m1.clone().to(dpcpp_device)
        m2_dpcpp_orig = m2.clone().to(dpcpp_device)
        acc_dpcpp_orig = acc.clone().to(dpcpp_device)
        model = MatmulSum()
        raw = model(m1, m2, acc)
        print("raw: ", raw)

        model = model.to(dpcpp_device)
        modelJit = torch.jit.script(model)

        m1_dpcpp = m1_dpcpp_orig.clone()
        m2_dpcpp = m2_dpcpp_orig.clone()
        acc_dpcpp = acc_dpcpp_orig.clone()
        #for i in range(2):
        #    modelJit(m1_dpcpp, m2_dpcpp, acc_dpcpp)

        with torch.no_grad():
            #if print_graph:
            #    print(modelJit.graph_for(m1_dpcpp, m2_dpcpp, acc_dpcpp))
            m1_dpcpp = m1_dpcpp_orig.clone()
            m2_dpcpp = m2_dpcpp_orig.clone()
            acc_dpcpp = acc_dpcpp_orig.clone()
            for i in range(10):
                real = modelJit(m1_dpcpp, m2_dpcpp, acc_dpcpp)
            print("real: ", real.cpu())
        #self.assertEqual(raw.shape, real.shape)
        #self.assertEqual(raw, real.to(cpu_device))
        del modelJit

m1 = torch.randn([6], device=cpu_device)
m2 = torch.randn([6], device=cpu_device)

#m1 = torch.randn([4, 2], device=cpu_device)
#m2 = torch.randn([2], device=cpu_device)

#m1 = torch.randn([2], device=cpu_device)
#m2 = torch.randn([2, 6], device=cpu_device)

#m1 = torch.randn([4, 2], device=cpu_device)
#m2 = torch.randn([2, 6], device=cpu_device)

#m1 = torch.randn([3, 4, 2], device=cpu_device)
#m2 = torch.randn([2, 6], device=cpu_device)
#m2 = torch.randn([2], device=cpu_device)

#m1 = torch.randn([2], device=cpu_device)
#m1 = torch.randn([6, 2], device=cpu_device)
#m2 = torch.randn([3, 2, 4], device=cpu_device)

#m1 = torch.randn([5, 1, 4, 2], device=cpu_device)
#m2 = torch.randn([3, 2, 4], device=cpu_device)

model = MatmulRelu()
matmul_fusion(model, m1, m2)

#shapes = [[[4, 2], [2], [4]]]
#test_matmul_sum_fusion(shapes)
