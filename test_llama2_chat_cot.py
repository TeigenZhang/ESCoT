from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import sys
import argparse
import torch

def main(args):
    prompt = "Generate the response using the pipeline of emotion, emotion stimulus, individual appraisal, strategy reason and response."

    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    model =  LlamaForCausalLM.from_pretrained(args.model_path, device_map=device, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print("Human:")
    line = input()
    inputs = ""
    while line:
        if inputs == "":
            inputs = 'Human: ' + line.strip() + '\n' + prompt + '\nAssistant:'
        else:
            inputs = inputs.replace("Human: ", "")
            inputs = inputs.replace('\n'+prompt, "")
            inputs = inputs + '\nseeker: ' + line.strip()
            inputs = 'Human: ' + inputs + '\n' + prompt + '\nAssistant:'

        input_ids = tokenizer(inputs, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        outputs = model.generate(input_ids, max_new_tokens=500, do_sample = True, top_k = 30, top_p = 0.85, temperature = 0.5, repetition_penalty=1., eos_token_id=2, bos_token_id=1, pad_token_id=0)
        rets = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = rets[0].strip().replace(inputs, "")

        inputs = inputs.replace("\nAssistant:", '\nsupporter:'+ response.strip())
        print("\nAssistant:" + response)
        print("\n------------------------------------------------\nSeeker:")
        line = input()

        if line == "clear":
            inputs = ""
            print("History cleared.")
            print("\n------------------------------------------------\nHuman:")
            line = input()
        if line == "exit":
            sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="The path of the pretrained model.")
    parser.add_argument("--gpu_id", type=int, required=True, help="The id of the GPU to be used.")
    args = parser.parse_args()

    main(args)
