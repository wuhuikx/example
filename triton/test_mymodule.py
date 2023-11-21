import torch
import intel_extension_for_pytorch

def my_compiler(gm: torch.fx,GraphMode):
    print("FX graph:")
    print(gm.graph)
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)
 
    def forward(self, x, y):
        a = torch.sin(x)
        b = torch.cos(x)
        return a + b
        #return torch.nn.functional.relu(self.lin(x))

mod = MyModule()
mod.eval()
mod = mod.xpu()
opt_foo = torch.compile(mod, backend=my_compiler)
#opt_foo = torch.compile(mod, backend="inductor")
print(opt_foo)

a = torch.randn(10, 100).xpu()
b = torch.randn(10, 100).xpu()
print("-----running-----")
c = opt_foo(a, b).cpu()
#print(opt_foo(a, b).cpu())
