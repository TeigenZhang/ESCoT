#!/bin/sh

export CUDA_VISIBLE_DEVICES=0,1,2,3

base_model='./pretrained_models/Llama-2-7b-chat-hf'
dataset_base='./data/ablation_data/'
output_base='./checkpoints/cot/'
lr_scheduler_type='cosine'
learning_rate=5e-5
max_steps=380
save_freq=38
eval_freq=380
seq_length=2048
batch_size=8

settings=("em_es_ia_sr_re" "em_sr_re" "sr_re" "re")

for setting in "${settings[@]}"
do
    run_name="supervised_llama2_cot_ablation_${setting}"
    
    dataset_name="${dataset_base}${setting}"

    output_dir="${output_base}${run_name}"

    echo "Running setting: $setting"
    echo "Dataset path: $dataset_name"
    echo "Output path: $output_dir"

    torchrun --nnodes 1 --nproc_per_node 4 supervised_finetuning_cot.py \
        --base_model "$base_model" \
        --dataset_name "$dataset_name" \
        --lr_scheduler_type "$lr_scheduler_type" \
        --learning_rate "$learning_rate" \
        --max_steps "$max_steps" \
        --save_freq "$save_freq" \
        --eval_freq "$eval_freq" \
        --seq_length "$seq_length" \
        --batch_size "$batch_size" \
        --run_name "$run_name" \
        --output_dir "$output_dir" \
        --save_total_limit 100 \
        --seed 1104
done
