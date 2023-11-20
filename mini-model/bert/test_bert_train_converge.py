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

# from schedulers import LinearWarmUpScheduler, LinearWarmupPolyDecayScheduler
# from lamb import Lamb

batch_size = 16
running_step = 15
datatype=torch.bfloat16
max_predictions_per_seq = 76
files="miniwiki/hdf5/pretrain-part-00.hdf5"
device="xpu"
max_steps_for_scheduler=1094400 
max_samples_termination=21012480
warmup_proportion=0.01
 
# get data
train_data = pretraining_dataset(files, max_predictions_per_seq)
train_dataloader = DataLoader(train_data, batch_size=batch_size)

# create model
config = AutoConfig.from_pretrained("bert_config.json")
model = AutoModelForPreTraining.from_config(config)

# model to device
model = model.to(device)
model.train()

# create optimizer and lr_scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# lr_scheduler = LinearWarmupPolyDecayScheduler(optimizer)
    
# enable ipex optimize for performance acceleration
model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, dtype=datatype)

# run training
    for step, batch in enumerate(train_dataloader):
        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
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
        torch.xpu.synchronize()
        end_time = time.time()
        if need_profile:
            prof.step()
        if step > running_step:
            break
        

    print("---latency={} s".format(end_time-start_time))
    print("---throughput={} fps".format(batch_size/(end_time-start_time)))
 

