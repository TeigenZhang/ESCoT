#!/bin/sh

export CUDA_VISIBLE_DEVICES=0,1,2,3

run_name='supervised_llama2_cot'

torchrun --nnodes 1 --nproc_per_node 4 supervised_finetuning_cot.py \
    --base_model '/datassd2/ztg/pretrained_models/Llama-2-7b-chat-hf' \
    --dataset_name './data/ablation_data/em_es_ia_sr_re' \
    --lr_scheduler_type 'cosine' \
    --learning_rate 1e-5 \
    --max_steps 10000 \
    --save_freq 500 \
    --seq_length 2048 \
    --batch_size 8 \
    --run_name $run_name \
    --output_dir './checkpoints/cot/'$run_name
