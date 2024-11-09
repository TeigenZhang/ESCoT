#!/bin/bash

# Define the model path and GPU id as variables
MODEL_PATH="change_to_checkpoint_path"
GPU_ID=0

# Execute the Python script with the model path and GPU id as arguments
python test_llama2_chat_cot.py --model_path $MODEL_PATH --gpu_id $GPU_ID
