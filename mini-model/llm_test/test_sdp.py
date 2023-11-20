import torch
import intel_extension_for_pytorch
import math


layer_id=0
prefix = "query_"
name = prefix + str(layer_id) + ".pt"
query = torch.load(name)
prefix = "key_"
name = prefix + str(layer_id) + ".pt"
key = torch.load(name)
prefix = "value_"
name = prefix + str(layer_id) + ".pt"
value = torch.load(name)
prefix = "attention_mask_"
name = prefix + str(layer_id) + ".pt"
attention_mask = torch.load(name)
prefix = "head_mask_"
name = prefix + str(layer_id) + ".pt"
head_mask = torch.load(name)

# llama config
first_token = False
greedy = True
use_casual_mask = False
head_dim = 128
attn_weights = None
blocked_alibi = None
max_positions = 2048

# parameters
dropout = 0.0
alpha = 1.0 / math.sqrt(head_dim)
beta = 1.0 # TODO: ignored by native

def get_blocked_attn_mask(attn_mask):
    blocked_attn_mask = torch.empty((attn_mask.shape[0], attn_mask.shape[1], attn_mask.shape[2], max_positions), device=attn_mask.device, dtype=attn_mask.dtype)
    blocked_attn_mask.fill_(-65504.);
    blocked_attn_mask[:, :, :, 0 : attn_mask.shape[3]] = attn_mask
    return blocked_attn_mask

is_causal = False
if use_casual_mask == True and query.shape[2] != 1:
    is_causal = True

blocked_attn_mask = None
if attention_mask != None:
    # transform the attention_mask to casual mask if the attention_mask is in bool
    if attention_mask.dtype == torch.bool:
        blocked_attn_mask = None
        if query.shape[2] != 1:
            is_causal = True
    else:
        blocked_attn_mask = get_blocked_attn_mask(attention_mask)

if first_token or greedy:
    seq_first = False 
    
    if greedy:
        seq_first = True
attn_output = torch.xpu.IpexSDP(query, key, value, blocked_alibi, blocked_attn_mask, head_mask, alpha, beta, dropout, is_causal, seq_first)