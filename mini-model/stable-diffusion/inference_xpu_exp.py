import os
import time
import torch
import intel_extension_for_pytorch as ipex
from diffusers import StableDiffusionPipeline
from diffusers import LMSDiscreteScheduler

profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]

dtype=torch.float
datatype=torch.float16
device="xpu"
seed = 666
idx = torch.xpu.current_device()
generator = torch.xpu.default_generators[idx]
generator.manual_seed(seed)
#generator = torch.Generator(device=device).manual_seed(seed)

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=datatype)
#pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
print("---pipe.scheduler.config={}".format(pipe.scheduler.config))
pipe = pipe.to(device)
'''
prompt = ["a photo of an astronaut riding a horse on mars",
          "a photo of a cat skates in INTEL Square",
          "a photo of Toothpaste with INTEL and shovel with AMD",
          "a poster of attractive girla poster of the pytorch xpu building"]
'''
prompt = ["a photo of an astronaut riding a horse on mars",
          "a photo of a cat skates in Square",
          "A painting of a squirrel eating a burger"]
prompt = ["a photo of an astronaut riding a horse on mars"]


# ipex.enable_auto_channels_last()
# pipe.unet = torch.xpu.optimize(model=pipe.unet, dtype=dtype)

# latent_model_input = torch.randn([2, 4, 64, 64]).to(dtype).to(device)
# prompt_embeds = torch.randn([2, 77, 768]).to(dtype).to(device)
# t = torch.randn([1]).to(torch.int).to(device)
# pipe.unet = torch.jit.trace(pipe.unet, (latent_model_input, t, prompt_embeds), check_trace=False, strict=False)
# pipe.unet = torch.jit.freeze(pipe.unet)

#pipe.unet = torch.xpu.optimize(model=pipe.unet, dtype=dtype)

batch_size = 1
with torch.no_grad():
    for i in range(len(prompt)):
        print("-------iteration------")
        with torch.autograd.profiler_legacy.profile(profiling, use_xpu=True, record_shapes=True) as prof:
            start_time = time.time()
            #with torch.xpu.amp.autocast(enabled=True, dtype=dtype):
            image = pipe(prompt[i],  generator=generator).images[0]
            torch.xpu.synchronize()
            end_time = time.time()
    
        if profiling:
            torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profile.pt')
            torch.save(prof.table(sort_by="id", row_limit=-1), 'profile_detailed.pt')
            prof.export_chrome_trace('./profile_trace.json')

        latency = end_time - start_time
        print("---latency={} s".format(latency))
        print("---throughput={} fps".format(batch_size/latency))
 
        name = str(i) + ".png"
        image.save(name)

print("---model={}".format(pipe))


