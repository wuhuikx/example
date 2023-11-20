import torch
import torch
from diffusers import StableDiffusionPipeline
from diffusers import LMSDiscreteScheduler

dtype=torch.float16
#dtype=torch.float
seed = 666
device="cuda"
generator = torch.Generator(device=device).manual_seed(seed)

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=dtype)
#pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
print("---pipe.scheduler.config={}".format(pipe.scheduler.config))
pipe = pipe.to(device)

#prompt = "a photo of an astronaut riding a horse on mars"
#image = pipe(prompt,  generator=generator).images[0]
#image.save("cuda_astronaut_rides_horse.png")    

prompt = ["a photo of an astronaut riding a horse on mars",
          "a photo of a cat skates in Square",
          "A painting of a squirrel eating a burger"]

#for i in range(len(prompt)):
i=2
image = pipe(prompt[i],  generator=generator).images[0]
name = str(i) + "_.png"
image.save(name)
