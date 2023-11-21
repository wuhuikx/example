import os
import torch
import time
import intel_extension_for_pytorch
from dataset import pretraining_dataset
from torch.utils.data import DataLoader

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForPreTraining,
    SchedulerType,
    get_scheduler,
    set_seed,
)

batch_size = 16
datatype=torch.bfloat16
need_profile=True
max_predictions_per_seq = 76
files="miniwiki/hdf5/pretrain-part-00.hdf5"
device="xpu"
 
# get data
train_data = pretraining_dataset(files, max_predictions_per_seq)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
running_step = 15 #len(train_dataloader)

# create model
config = AutoConfig.from_pretrained("bert_config.json")
model = AutoModelForPreTraining.from_config(config)

# model to device
model = model.to(device)
model.train()

# create optimizer and lr_scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# enable ipex optimize for performance acceleration
model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, dtype=datatype)

# add profiler
def trace_handle(prof):
    print("using trace")
    print(str(prof.key_averages().table(sort_by="self_xpu_time_total")))
    torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profiling.pt')
    prof.export_chrome_trace('./profile_trace.json')

import contextlib
def profiler_setup(need_profile):
    print("---need_profile={}".format(need_profile))
    if need_profile:
        return torch.profiler.profile(
            activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU],
            schedule=torch.profiler.schedule(
            skip_first=running_step-5,
            wait=1,
            warmup=3,
            active=1),
            #on_trace_ready=trace_handle,
            record_shapes=True,
            profile_memory=False,  
            with_stack=True
        )
    else:
        return contextlib.nullcontext()

# run training   
with profiler_setup(need_profile) as prof:
    for step, batch in enumerate(train_dataloader):
        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
        start_time = time.time()
        print("input_ids={}".format(input_ids))
        print("segment_ids={}".format(segment_ids))
        print("input_mask={}".format(input_mask))
        print("masked_lm_labels={}".format(masked_lm_labels))
        print("next_sentence_labels={}".format(next_sentence_labels))
        # input data H2D
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
        loss = outputs.loss
        loss.backward()

        # weight update
        optimizer.step()
        optimizer.zero_grad()
        torch.xpu.synchronize()
        end_time = time.time()
        if step == running_step:
            break
        if need_profile:
            prof.step()
        
    if need_profile:
        print(str(prof.key_averages().table(sort_by="self_xpu_time_total")))
        torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profiling.pt')
        prof.export_chrome_trace('./profile_trace.json')

    print("---latency={} s".format(end_time-start_time))
    print("---throughput={} fps".format(batch_size/(end_time-start_time)))
 

