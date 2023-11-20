import os
import torch
import time
import intel_extension_for_pytorch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
)

profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
batch_size = 64 
datatype=torch.float16
max_sequence_length = 384
device="xpu"

# step1. prepare
config_name="./bert_config.json"
model_name_or_path = "/media/storage/wuhui/gpu-models/Bert-MLPerf/squad/cached_dev_squad_large_finetuned_checkpoint_384"
config = AutoConfig.from_pretrained(config_name)
model = AutoModelForQuestionAnswering.from_pretrained(
    model_name_or_path,
    from_tf=bool(".ckpt" in model_name_or_path),
    config=config)
model = model.to(device)
model.eval()

# step2. enable ipex optimize
model = torch.xpu.optimize(model=model, dtype=datatype)

input_ids = torch.randint(10, 380, (batch_size, max_sequence_length)).xpu()
token_type_ids = torch.randint(10, 380, (batch_size, max_sequence_length)).xpu()
attention_mask = torch.randint(10, 380, (batch_size, max_sequence_length)).xpu()
# step3. enable jit + amp
with torch.no_grad():
    with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
        modelJit = torch.jit.trace(model, (input_ids, token_type_ids, attention_mask), strict=False)
        modelJit = torch.jit.freeze(modelJit)

warmup_iter = 5
total_time = 0
# step4. do inference
with torch.no_grad():
    for step in range(20):
        input_ids = torch.randint(10, 380, (batch_size, max_sequence_length))
        token_type_ids = torch.randint(10, 380, (batch_size, max_sequence_length))
        attention_mask = torch.randint(10, 380, (batch_size, max_sequence_length))

        # insert profiling (only for development)
        with torch.autograd.profiler_legacy.profile(profiling, use_xpu=True, record_shapes=True) as prof:
            start_time = time.time()
            # H2D
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)

            # forward
            outputs = modelJit(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask)

            # D2H
            for k, v in outputs.items():
                v = v.to(torch.float32).to("cpu")

            # sync
            torch.xpu.synchronize()
            end_time = time.time()

            if step >= warmup_iter:
                total_time += end_time - start_time
                latency = total_time / (step - warmup_iter + 1)
                throughput = batch_size / latency
                print("---latency={} ms".format(latency))
                print("---throughput={} fps".format(throughput))

        if profiling:
            with open('./profile.pt', 'w') as f:
                f.write(str(prof.key_averages().table(sort_by="self_xpu_time_total")))
            with open('./profile_detailed.pt', 'w') as f1:
                f1.write(str(prof.table(sort_by="id", row_limit=-1)))
            prof.export_chrome_trace('./profile_trace.json')

        if step + 1 >= 10:
            break

