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




from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

if __name__ == "__main__":
    main()
