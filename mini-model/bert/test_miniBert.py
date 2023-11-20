import os
import torch
import time
import intel_extension_for_pytorch
from dataset import pretraining_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

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

batch_size = 1
datatype=torch.bfloat16
profile=False
max_predictions_per_seq = 76
files="../../Bert-MLPerf/pretrain_mlperf/data/hdf5_seq_512/part-00001-of-00500.hdf5"
device="xpu"

# get data
train_data = pretraining_dataset(files, max_predictions_per_seq)
train_dataloader = DataLoader(train_data, batch_size=batch_size)

# setup model
config_name="bert_config.json"
config = AutoConfig.from_pretrained(config_name)
model = AutoModelForPreTraining.from_config(config)
model = model.to(device)
model.train()

# setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# enable ipex optimize for performance acceleration
model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, dtype=datatype)

# run training
for step, batch in enumerate(train_dataloader):
    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch

    # insert profiling (only for development)
    with torch.autograd.profiler_legacy.profile(profile, use_xpu=True) as prof:
        start_time = time.time()
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

        # gradient weight normalization
        if hasattr(optimizer, "clip_grad_norm_"):
            ggnorm = optimizer.clip_grad_norm_(1.0)
        else:
            ggnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # weight update
        optimizer.step()
        optimizer.zero_grad()
        end_time = time.time()

    if profile:
        print(str(prof.key_averages().table(sort_by="self_xpu_time_total")))
        with open('./profile_trace.txt', 'w') as f:
            f.write(str(prof.table(sort_by="id", row_limit=100000)))
        prof.export_chrome_trace('./profile_trace.json')

    print("---latency={} ms".format(end_time-start_time))
    print("---throughput={} fps".format(batch_size/(end_time-start_time)*1000))
    if step > 20:
        break

