""""""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import json

# Specify the model you want to use
# model_name = "meta-llama/Llama-2-70b-chat-hf"
model_name = "epfl-llm/meditron-7b"

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
    device_map="auto",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('questions.json') as json_file:
    questions = json.load(json_file)
system_prompt = "<s>< |im_start| >system\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n< |im_end| >\n< |im_start| > question\n"

def main(n_sample: int = 5, max_i_q: int = 3, debug_verbose: bool = True):  
    """"""

    with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['Category', 'Q_Set'] + [f'Reply{i+1}' for i in range(max_i_q)]
        csvwriter.writerow(header)

        for category in questions:
            for q_set in questions[category]:
                all_samples_replies = [[] for _ in range(n_sample)]
                contexts = [""] * n_sample
                first_question = True

                for q in questions[category][q_set]:
                    for i in range(n_sample):
                        print(f"---- \n {category} {q_set} {i} \n ---- ")
                        if first_question:
                            contexts[i] += system_prompt
                        full_prompt = contexts[i] + q + " \n< |im_start| >answer\n"
                        prompt_length = len(tokenizer.tokenize(full_prompt))
                        total_max_length = prompt_length + 300

                        input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)

                        if debug_verbose:
                            print(i)
                            print("Input Text")
                            print(full_prompt)
                            print()
                        
                        # Generate text
                        outputs = model.generate(
                            input_ids,
                            max_length=total_max_length,
                            pad_token_id=tokenizer.eos_token_id,
                            do_sample=True,
                            top_k=10,
                            num_return_sequences=1
                        )

                        # Decode the generated text
                        # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        clean_seq = tokenizer.decode(outputs[0, (prompt_length + 1):])    
                        all_samples_replies[i].append(clean_seq)

                        # Update context
                        contexts[i] += f"{q} < |im_end| >\n< |im_start| >answer\n{clean_seq} < |im_end| ></s><s>< |im_start| > question\n"

                    if first_question:
                        first_question = False

                if debug_verbose:
                    for c in contexts:
                        print(c)

                for replies in all_samples_replies:
                    row = [category, q_set] + replies[:max_i_q]
                    csvwriter.writerow(row)


if __name__ == "__main__":
    main()
