import torch
import intel_extension_for_pytorch
import math

# llama config
batch_size = 1
beam_size = 1
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
inv_norm_factor = 1.0 / math.sqrt(head_dim)
scale_attn = torch.sqrt(torch.tensor(head_dim, device="xpu"))


layer_id=7  #diff=0.02001953125
#layer_id=13  #diff=0.02783203125
#layer_id=17  #diff=0.023193359375
#layer_id=18  #diff=0.0224609375
#layer_id=19  #diff=0.0212860107421875
#layer_id=21  #diff=0.036376953125
#layer_id=22  #diff=0.0301513671875
#layer_id=23  #diff=0.028564453125
#layer_id=24  #diff=0.027587890625
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

def naive_self_attention(query, key, value, attention_mask=None, head_mask=None, alibi : torch.Tensor=None, first_token=False):
    if alibi is not None:
        bs_beam, num_heads, q_length, dim = query.shape
        _, _, kv_length, _ = key.shape
        # query, key result [bs*beam, num_head, q_len, kv_len]
        # alibi: [bs_beam*num_head, q_len, kv_len]
        if first_token and not greedy:
            shape = [batch_size, beam_size, num_heads, -1, kv_length]
            alibi = alibi.view(shape)[:, 0, :, :, :].reshape([batch_size*num_heads, -1, kv_length])
        batch1 = query.view(-1, q_length, dim)
        batch2 = key.view(-1, kv_length, dim).transpose(1, 2)
        matmul_result = alibi.baddbmm(
            batch1=batch1,
            batch2=batch2,
            beta=beta,
            alpha=inv_norm_factor,
        )

        # change view to [bs_beam, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(bs_beam, num_heads, q_length, kv_length)
        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = torch.nn.functional.softmax(attn_weights, dim=-1)

        # [bs_beam, num_heads, q_length, kv_length]
        #attention_probs = attn_drop(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # matmul: [bs_beam * num_heads, q_length, head_dim]
        attn_output = torch.matmul(attention_probs, value)
    else:
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if use_casual_mask:
            # convert the casual mask dtype to target dtype, this should only happen once
            attention_mask.to(attn_weights.dtype)
            query_length, key_length = query.size(-2), key.size(-2)
            casual_mask = attention_mask[:, :, key_length - query_length : key_length, :key_length]
            # # TODO: Maybe we can move this line to the initializer
            # casual_mask *= -66504.0
            # replace torch.where as torch.add might helps with the host overhead
            attn_weights += casual_mask
        if scale_attn:
            attn_weights /= scale_attn
        if attention_mask is not None:
            attn_weights += attention_mask
            # the attn_weights should anyway bigger than dtype.min, I wonder if this is necessary
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float).to(query.dtype)
        #attn_weights = attn_drop(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights



def get_blocked_attn_mask(attn_mask):
    blocked_attn_mask = torch.empty((attn_mask.shape[0], attn_mask.shape[1], attn_mask.shape[2], max_positions), device=attn_mask.device, dtype=attn_mask.dtype)
    blocked_attn_mask.fill_(-65504.)
    blocked_attn_mask[:, :, :, 0 : attn_mask.shape[3]] = attn_mask
    return blocked_attn_mask

def optimize_sdp(key_prompt, value_prompt, query, key, value, attention_mask=None, head_mask=None, alibi=None, first_token=False):
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
    return attn_output

attn_ref, _ = naive_self_attention(query, key, value, attention_mask, head_mask, alibi=None, first_token=False)
key_prompt = None
value_prompt = None
attn_opt = optimize_sdp(key_prompt, value_prompt, query, key, value, attention_mask, head_mask, alibi=None, first_token=False)

diff = torch.max(torch.abs(attn_ref - attn_opt))
print("diff={}".format(diff))