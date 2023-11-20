import os
import torch
import time
import intel_extension_for_pytorch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForPreTraining,
)

profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
batch_size = 16
datatype=torch.bfloat16
max_predictions_per_seq = 76
device="xpu"
sequence_length = 512

# step1. prepare
config_name="./bert_config.json"
#config_name="/media/storage/wuhui/gpu-models/mini-model/bert/bert_config.json"
config = AutoConfig.from_pretrained(config_name)
model = AutoModelForPreTraining.from_config(config)
model = model.to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# step2. enable ipex optimize
model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, dtype=datatype)

# step3. do training
for step in range(50):
    # import sys
    # sys.stdout.flush()
    input_ids = torch.randint(10, 510, (batch_size, sequence_length))
    segment_ids = torch.randint(10, 100, (batch_size, sequence_length))
    input_mask = torch.randint(10, 100, (batch_size, sequence_length))
    masked_lm_labels = torch.randint(10, 100, (batch_size, sequence_length))
    next_sentence_labels = torch.randint(10, 100, (batch_size,))

    # insert profiling (only for development)
    with torch.autograd.profiler_legacy.profile(profiling, use_xpu=True, record_shapes=True) as prof:
    # with torch.autograd.profiler.emit_itt():
    #     torch.profiler.itt.range_push('step_{}'.format(step))
        start_time = time.time()
        # H2D
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        input_mask = input_mask.to(device)
        masked_lm_labels = masked_lm_labels.to(device)
        next_sentence_labels = next_sentence_labels.to(device)

        # enable amp for low precision training (bfloat16)
        with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
            # forward
            outputs = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=masked_lm_labels,
                next_sentence_label=next_sentence_labels)

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss = outputs.loss
        loss.backward()

        # gradient weight normalization
        if hasattr(optimizer, "clip_grad_norm_"):
            ggnorm = optimizer.clip_grad_norm_(1.0)
        else:
            ggnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # weight update
        optimizer.step()
        torch.xpu.synchronize()
        # torch.profiler.itt.range_pop()
        end_time = time.time()

    if profiling:
        torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profile.pt')
        torch.save(prof.table(sort_by="id", row_limit=-1), 'profile_detailed.pt')
        prof.export_chrome_trace('./profile_trace.json')

    latency = end_time-start_time
    print("---latency={} s".format(latency))
    print("---throughput={} fps".format(batch_size/latency))
    if step > 20:
        break

