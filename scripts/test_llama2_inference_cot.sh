#!/bin/bash

# Define the model path and GPU id as variables
MODEL_PATH="change_to_checkpoint_path"
JSON_PATH="./data/ablation_data/em_es_ia_sr_re/empathetic_dialogue_valid.json"
GPU_ID=0

# Execute the Python script with the model path and GPU id as arguments
python test_llama2_inference_cot.py --model_path $MODEL_PATH --gpu_id $GPU_ID --json_path $JSON_PATH
