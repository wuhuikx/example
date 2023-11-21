import torch
import time
import numpy as np
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")

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
debug = True
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger()
logger.info("This is for bert pre-training")
# stream_handler = logging.StreamHandler(sys.stdout)
# logger.addHandler(stream_handler)

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
if debug:
    print(config)
    print(model)

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
    print("----flag1-----")
    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
    start_time = time.time()
    
    # input_ids # [16, 512], int64, token_id
    # segment_ids #[16, 512], int64, 0 and 1 for two sentences
    # input_mask  #[16, 512], int64, 1 and 0 for real data and padding data
    # masked_lm_labels # [16, 512], int64, the masked position with real token_id
    # next_sentence_labels # [16], int64, two sentences in a batch is successive or not    
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
    if step > running_step:
        break
    

# logging.debug("latency={} s".format(end_time-start_time))
# logging.debug("throughput={} fps".format(batch_size/(end_time-start_time)))
 

