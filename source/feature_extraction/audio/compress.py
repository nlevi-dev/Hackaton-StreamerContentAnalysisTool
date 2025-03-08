import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

CONTEXT_L = 10*60
CONTEXT_S = 2*60
STEP = 60

raw = np.load("/mnt-persist/test/1/Our_New_4500_Workstation_PCs_for_Editing_audio_raw.npy")
END = int(raw[-1,1])

PROMPT_L = "========================\nBased on this broader context of a conversation:\n========================\n"
PROMPT_S = "========================\nBased on this recent context of a conversation:\n========================\n"
PROMPT = "========================\nAnswer the following question, related to the recent context from the provided conversation:\n========================\n"

chunks = []
for i in range(END//STEP):
    end_s = i*STEP+STEP
    start_s = end_s-CONTEXT_S
    end_l = start_s
    start_l = end_s-CONTEXT_L
    context_l = ""
    context_s = ""
    for s in raw:
        st = int(s[0])
        if start_l <= st and st < end_l:
            context_l += s[3]+"\n"
        elif start_s <= st and st <= end_s:
            context_s += s[3]+"\n"
    chunk = ""
    if len(context_l) > 0:
        chunk += PROMPT_L
        chunk += context_l
    chunk += PROMPT_S
    chunk += context_s
    chunk += PROMPT
    chunks.append(chunk)

print(len(chunks))
print(chunks[50])

torch.random.manual_seed(0)
model_path = "microsoft/Phi-4-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
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
result = pipe(prompt, **generation_args)[0]['generated_text']
