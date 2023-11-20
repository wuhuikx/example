import torch
import intel_extension_for_pytorch

print("=========FWD=======")
# out = beta * input + alpha * (batch1 @ batch2)
M = torch.randn(3, 5).to("xpu")
batch1 = torch.randn(10, 3, 4).to("xpu")
batch2 = torch.randn(10, 4, 5).to("xpu")
M.requires_grad_()
batch1.requires_grad_()
batch2.requires_grad_()
out = torch.addbmm(M, batch1, batch2, beta=3, alpha=2)

print("=========BWD=======")
grad = torch.randn(out.shape).to("xpu")
out.backward(grad)
