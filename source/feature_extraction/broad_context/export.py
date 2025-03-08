import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model_path = "microsoft/Phi-4-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a simple "Hello, World!" prompt
prompt = "Hello, World!"

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 50,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(prompt, **generation_args)
print(output[0]['generated_text'])
