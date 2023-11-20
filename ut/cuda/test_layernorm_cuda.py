import torch
import torch.nn as nn

batch, sentence_length, embedding_dim = 16, 512, 1024
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
# Activate module

embedding_cuda = embedding.to("cuda")
layer_norm_cuda = layer_norm.to("cuda")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
       wait=1,
       warmup=1,
       active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./ut_layernorm', worker_name='worker0'),
    record_shapes=True,
    profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    with_stack=True) as p:
    for i in range(5):
        y = layer_norm_cuda(embedding_cuda)
        torch.cuda.synchronize()
        p.step()

# print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

