import torch 
import intel_extension_for_pytorch

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 10)
        self.linear = torch.nn.Linear(10, 10)
        self.lm_head = torch.nn.Linear(10, 10)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        x = self.lm_head(x)
        return x

model = DummyModel()
model = model.xpu()


activities = [torch.profiler.ProfilerActivity.CPU]
activities.append(torch.profiler.ProfilerActivity.XPU)

def trace_handle(prof):
    print("using trace")
    print(str(prof.key_averages().table(sort_by="self_xpu_time_total")))
    torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profiling.pt')
    prof.export_chrome_trace('./profile_trace.json')

schedule = torch.profiler.schedule(skip_first=5, wait=1, warmup=3, active=1)
with torch.profiler.profile(
    activities=activities,
    schedule=schedule,
    on_trace_ready=trace_handle,
) as prof:
    for i in range(10):
        dummy_inputs = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        dummy_inputs = dummy_inputs.to("xpu")
        dummy_outputs = model(dummy_inputs)
        prof.step()
#print("using trace")
#print(str(prof.key_averages().table(sort_by="self_xpu_time_total")))
#torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profiling.pt')

