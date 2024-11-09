import json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torch
from tqdm import tqdm
import os

# Function to prepare data
def prepare_sample_text(data_point):
    """Prepare the text from a sample of the dataset."""
    return f"""<s>Human: 
{data_point["input"]}
{data_point["instruction"]}
</s><s>Assistant: """
    
# Main function
def main(args):
    # Load model and tokenizer
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    model = LlamaForCausalLM.from_pretrained(args.model_path, device_map=device, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Read JSON file
    with open(args.json_path, 'r') as file:
        data = json.load(file)

    # Store inference results
    results = []

    # Perform inference for each data point
    for data_point in tqdm(data):
        prepared_text = prepare_sample_text(data_point)
        input_ids = tokenizer(prepared_text, return_tensors="pt").input_ids
        prepared_text = tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        input_ids = input_ids.to(device)

        # Execute model inference
        outputs = model.generate(input_ids, max_new_tokens=500, do_sample=True, top_k=30, top_p=0.85, temperature=0.5, repetition_penalty=1., eos_token_id=2, bos_token_id=1, pad_token_id=0)
        rets = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # Extract response and remove original text
        full_response = rets[0].strip()
        response = full_response.replace(prepared_text, "").strip()

        # Save results
        results.append({'input': data_point['input'], 'label': data_point['output'], 'prediction': response, 'dialog_id': data_point['dialog_id']})

        # Save results to file
        if "test" in args.json_path:
            with open(os.path.join(args.model_path, 'test_inference_results.json'), 'w') as outfile:
                json.dump(results, outfile, indent=4, ensure_ascii=False)
        else:
            with open(os.path.join(args.model_path, 'val_inference_results.json'), 'w') as outfile:
                json.dump(results, outfile, indent=4, ensure_ascii=False)

# Use argparse to handle command line arguments
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="The path of the pretrained model.")
    parser.add_argument("--gpu_id", type=int, required=True, help="The id of the GPU to be used.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file containing the data.")
    
    args = parser.parse_args()

    main(args)