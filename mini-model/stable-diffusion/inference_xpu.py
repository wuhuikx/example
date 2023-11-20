import os
import time
import sys
import torch
from diffusers import StableDiffusionPipeline
import argparse

parser = argparse.ArgumentParser(description='PyTorch StableDiffusion TexttoImage')
parser.add_argument("--arch", type=str, default='CompVis/stable-diffusion-v1-4', help="model name")
parser.add_argument('--prompt', default=[
    "a photo of an astronaut riding a horse on mars", 
    "a photo of a cat skates in Square", 
    "A painting of a squirrel eating a burger",
    "A photo of dog in the room"], type=list, help='prompt')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--idx_start', default=0, type=int, help='select the start index of image')
parser.add_argument('--precision', default="float32", type=str, help='precision')
parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
parser.add_argument('--iteration', default=5, type=int, help='test iterations')
parser.add_argument('--warmup_iter', default=2, type=int, help='test warmup')
parser.add_argument('--device', default='xpu', type=str, help='cpu, cuda or xpu')
parser.add_argument('--save_image', action='store_true', default=False, help='save image')
parser.add_argument('--accuracy', action='store_true', default=False, help='compare the result with cuda')
parser.add_argument('--ref_path', default='', type=str, metavar='PATH',
                    help='path to reference image (default: none)')
parser.add_argument('--num_inference_steps', default=50, type=int, help='number of unet step')
args = parser.parse_args()
print(args)

def compare(xpu_res, ref_res):
    xpu = torch.tensor(xpu_res) 
    ref = torch.tensor(ref_res) 
    
    diff_value = torch.abs((xpu - ref))
    max_diff = torch.max(diff_value)

    shape = 1
    for i in range(xpu.dim()):
        shape = shape * xpu.shape[i]

    value = diff_value > 0.1
    num = torch.sum(value.contiguous().view(-1))
    ratio1 = num / shape
    print("difference larger than 0.1, ratio = {}".format(ratio1))  
  
    value = diff_value > 0.01 
    num = torch.sum(value.contiguous().view(-1))
    ratio2 = num / shape
    print("difference larger than 0.01, ratio = {}".format(ratio2))  

    value = diff_value > 0.001 
    num = torch.sum(value.contiguous().view(-1))
    ratio3 = num / shape
    print("difference larger than 0.001, ratio = {}".format(ratio3))  

    if ratio1 < 0.01 and ratio2 < 0.08 and ratio3 < 0.4:
        print("accuracy pass")
    else:
        print("accuracy fail")
   
def main():
    profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]

    # prompt = ["A painting of a squirrel eating a burger"]
    seed = 666
    if args.device == "xpu":
        import intel_extension_for_pytorch as ipex
        idx = torch.xpu.current_device()
        generator = torch.xpu.default_generators[idx]
        generator.manual_seed(seed)
    elif args.device == "cuda":
        generator = torch.Generator(device=args.device).manual_seed(seed)
    else:
        generator = torch.Generator(device=args.device)

    if args.precision == "fp32":
        datatype = torch.float
    elif args.precision == "fp16":
        datatype = torch.float16
    elif args.precision == "bf16":
        datatype = torch.bfloat16
    else:
        print("unsupported datatype")
        sys.exit()

    if args.device == "xpu":
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe = pipe.to(args.device)
        pipe.unet = torch.xpu.optimize(model=pipe.unet, dtype=datatype)
        # pipe = torch.jit.trace(pipe, input, strict=False)
    else:
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=datatype, revision=args.precision)
        pipe = pipe.to(args.device)

    out_type = "pil"
    if args.accuracy:
        out_type = "tensor"

    total_time = 0
    with torch.no_grad():
        for step in range(args.iteration + args.warmup_iter):
            if step < args.warmup_iter:
                input = args.prompt[0: args.batch_size]
            else:
                idx1 = args.idx_start + int(step * args.batch_size)
                idx2 = args.idx_start + int((step + 1) * args.batch_size)
                print("--idx1={}".format(idx1))
                print("--idx2={}".format(idx2))
                input = args.prompt[idx1:idx2]

            if step >= args.warmup_iter:
                print("Iteration = {}".format(step - args.warmup_iter))

            if args.device == "xpu":
                with torch.autograd.profiler_legacy.profile(profiling, use_xpu=True, record_shapes=True) as prof:
                    start_time = time.time()
                    with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
                        images = pipe(input, generator=generator, num_inference_steps=args.num_inference_steps, output_type=out_type).images
                    torch.xpu.synchronize()
                    end_time = time.time()
                if profiling:
                    torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profile.pt')
                    torch.save(prof.table(sort_by="id", row_limit=-1), 'profile_detailed.pt')
                    prof.export_chrome_trace('./profile_trace.json')
            elif args.device == "cuda":
                start_time = time.time()
                images = pipe(input, generator=generator, num_inference_steps=args.num_inference_steps).images
                torch.cuda.synchronize()
                end_time = time.time()
            else:
                start_time = time.time()
                images = pipe(input, generator=generator, num_inference_steps=args.num_inference_steps).images
                end_time = time.time()


            if step >= args.warmup_iter:
                iter_time = end_time - start_time
                total_time += iter_time
                latency = total_time / (step - args.warmup_iter + 1)
                throughput = args.batch_size / latency
                # print("---latency={} s".format(latency))
                # print("---throughput={} fps".format(throughput))

            if args.accuracy:
                for i in range(args.batch_size):
                    name = "result_" + str(idx1 + i) +".pt"
                    torch.save(images[i], name)
                    name = os.path.join(args.ref_path, name);
                    cuda_image = torch.load(name)
                    compare(images[i], cuda_image)

            if args.save_image:
                for i in range(args.batch_size):
                    name = "result_" + str(idx1 + i) +".png"
                    torch.save(images[i], name)
       
if __name__ == '__main__':
    main() 
