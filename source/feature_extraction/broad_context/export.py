# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# torch.random.manual_seed(0)

# model_path = "microsoft/Phi-4-mini-instruct"

# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map="auto",
#     torch_dtype="auto",
#     trust_remote_code=True,
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# # Create a simple "Hello, World!" prompt
# prompt = "Hello, World!"

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
# )

# generation_args = {
#     "max_new_tokens": 50,
#     "return_full_text": False,
#     "temperature": 0.0,
#     "do_sample": False,
# }

# output = pipe(prompt, **generation_args)
# print(output[0]['generated_text'])




from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def process_batch(pipe, prompts, system_message):
    messages_list = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        text = pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        messages_list.append(text)

    generation_args = {
        "max_new_tokens": 512,
        "return_full_text": False,
        "batch_size": len(prompts)
    }

    responses = pipe(messages_list, **generation_args)
    return [r["generated_text"] for r in responses]

def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    system_message = "You are a data processing agent. Your task is to follow the instructions and extract the data from broader context in the exact form that the user requests it."
    
    # Example batch of prompts
    prompts = [
        "Give me a short introduction to large language models.",
        "Explain what transformers are in AI.",
        "What is transfer learning?"
    ]

    responses = process_batch(pipe, prompts, system_message)
    for prompt, response in zip(prompts, responses):
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")

if __name__ == "__main__":
    main()
