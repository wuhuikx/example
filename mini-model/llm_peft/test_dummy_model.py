import torch 
import intel_extension_for_pytorch
from peft import inject_adapter_in_model, LoraConfig


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 10)
        self.linear = torch.nn.Linear(10, 1024)
        self.lm_head = torch.nn.Linear(1024, 10)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        x = self.lm_head(x)
        return x

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    target_modules=["linear", "embedding"],
)

model = DummyModel()
model = inject_adapter_in_model(lora_config, model)
print(model)
model = model.xpu()

dummy_inputs = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])
dummy_inputs = dummy_inputs.to("xpu")


def trace_handle(prof):
    print("using trace")
    print(str(prof.key_averages().table(sort_by="self_xpu_time_total")))
    torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profiling.pt')
    prof.export_chrome_trace('./profile_trace.json')


with torch.profiler.profile(
    activities=[
       torch.profiler.ProfilerActivity.CPU,
       torch.profiler.ProfilerActivity.XPU],
    schedule=torch.profiler.schedule(
       skip_first=5,
       wait=1,
       warmup=3,
       active=1),
    on_trace_ready=trace_handle,
    #record_shapes=True,
    #profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    with_stack=True
) as prof:
    for i in range(10):
        dummy_outputs = model(dummy_inputs)
        prof.step()
# print(str(prof.key_averages().table(sort_by="self_xpu_time_total")))
# torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profiling.pt')
# print(str(prof.key_averages(group_by_input_shape=True).table(sort_by="self_xpu_time_total", row_limit=200000)))
#prof.export_chrome_trace('./profile_trace.json')
