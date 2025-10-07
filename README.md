# Tevatron

This repository modifies the tevatron (promptriever version) by adding a regularization term.

## Installation

```bash
pip install -e .

cd src/tevatron/retriever
```

## Training

### Repllama
```bash
nohup bash -c 'deepspeed --include localhost:"0,1,2,3" --master_port "60001" train.py \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir repllama-1B-no_weight \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --lora \
  --lora_r 16 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 2000 \
  --dataset_name "deu05232/promptriever-ours-v8-repllama" \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 16 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 304 \
  --passage_max_len 196 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --seed 42 \
  --data_seed 42 \
  --gradient_accumulation_steps 2 \
  --dataset_cache_dir /workspace/cache \
  --dont_shuffle \
  --negatives_first_n 3' > logs/train.log 2>&1 &
```

### Promptriever
```bash
nohup bash -c 'deepspeed --include localhost:"0,1,2,3" --master_port "60001" train.py \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir promptriever-1B-no_weight \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --lora \
  --lora_r 16 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 4000 \
  --dataset_name "deu05232/promptriever-ours-v8-vanilla" \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 16 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 304 \
  --passage_max_len 196 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --seed 42 \
  --data_seed 42 \
  --gradient_accumulation_steps 2 \
  --dataset_cache_dir /workspace/cache \
  --dont_shuffle \
  --negatives_first_n 3' > logs/train.log 2>&1 &
```

### w/ Regularization
```bash
nohup bash -c 'deepspeed --include localhost:"0,1,2,3" --master_port "60001" train.py \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir promptriever-1B-idea3-reduced_KL \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --lora \
  --lora_r 16 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 4000 \
  --dataset_name "deu05232/promptriever-ours-v8-vanilla" \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 16 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 304 \
  --passage_max_len 196 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --seed 42 \
  --data_seed 42 \
  --gradient_accumulation_steps 2 \
  --dataset_cache_dir /workspace/cache \
  --dont_shuffle \
  --negatives_first_n 3' > logs/train.log 2>&1 &


  nohup bash -c 'deepspeed --include localhost:"0,1,2,3" --master_port "60001" train.py \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir fixed_promptriever-1B-idea3-MSE \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --lora \
  --lora_r 16 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 250 \
  --dataset_name "deu05232/promptriever-ours-v8-vanilla" \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 16 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 304 \
  --passage_max_len 196 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --seed 42 \
  --data_seed 42 \
  --gradient_accumulation_steps 2 \
  --dataset_cache_dir /workspace/cache \
  --dont_shuffle \
  --negatives_first_n 3' > logs/train.log 2>&1 &
```
