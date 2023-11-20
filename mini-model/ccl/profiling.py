
import torch
import intel_extension_for_pytorch
import torch.distributed as dist
import os

profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

backend = 'ccl'
dist.init_process_group(backend)
my_rank = dist.get_rank()
my_size = dist.get_world_size()
print("my rank = %d  my size = %d" % (my_rank, my_size))

x = torch.ones([2, 2]).to("xpu")
y = torch.ones([4, 4]).to("xpu")
#with torch.autograd.profiler.profile(record_shapes=True) as prof:
with torch.autograd.profiler_legacy.profile(profiling, use_xpu=True) as prof:
    for _ in range(10):
        dist.all_reduce(x)
        dist.all_reduce(y)
dist.barrier()
print(prof.table(sort_by="id", row_limit=-1))

