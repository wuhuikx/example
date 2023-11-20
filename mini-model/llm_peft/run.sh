export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export ENABLE_SDP_FUSION=1
export ZE_AFFINITY_MASK=0.1

export HF_HOME=/workspace/huggingface/
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

python finetune.py  \
    --fp32 \
    --base_model 'meta-llama/Llama-2-7b-hf'\
    --data_path 'alpaca_data.json' \
    --output_dir './result' \
    --batch_size 32 \
    --micro_batch_size 32 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --max_steps 50 
