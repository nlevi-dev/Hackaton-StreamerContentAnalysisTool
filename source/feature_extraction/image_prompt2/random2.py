import os
import sys
import re
import _pickle as pickle
import numpy as np
import pandas as pd
import jaro
import torch
import torch.profiler
from PIL import Image
import requests
import transformers
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# Uncomment if using command-line arguments
if len(sys.argv) < 2:
    print("Usage: extract.py path_to_folder [--debug]")
    sys.exit(1)
path = sys.argv[1]
debug = "--debug" in sys.argv

def pickle_load(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

def pickle_save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def text_save(path, txt):
    with open(path, 'w') as f:
        f.write(txt)

def get_closest_idx(items, candidate):
    m = 0
    ret_idx = 0
    idx = 0
    for item in items:
        dist = jaro.jaro_winkler_metric(item, candidate)
        if dist > m:
            m = dist
            ret_idx = idx
        idx += 1
    return ret_idx

def generate_onehot(items):
    preprompt = "Answer from the following list with only using a word from it [" + ",".join(items) + "]!"
    closest = lambda a: get_closest_idx(items, a)
    return (preprompt, closest)

preprompts = {
    "list[str]": (
        "Answer with a short, comma separated list!",
        lambda a: [re.sub("[\\s]+", "_", b.strip().lower()) for b in a.split(',') if b.strip() != '']
    ),
    "list[int]": (
        "Answer with a short, comma separated list!",
        lambda a: [int(re.sub("[^\\d]", "", b)) for b in a.split(',') if len(b.strip()) > 0]
    ),
    "list[float]": (
        "Answer with a short, comma separated list!",
        lambda a: [float(re.sub("[^\\d\\.]", "", b)) for b in a.split(',') if b.strip() != '']
    ),
    "int": (
        "Answer with a single integer number!",
        lambda a: int(re.sub("[^\\d]", "", a)) if len(re.sub("[^\\d]", "", a)) > 0 else 0
    ),
    "float": (
        "Answer with a single float number!",
        lambda a: float(re.sub("[^\\d\\.]", "", a)) if len(re.sub("[^\\d\\.]", "", a)) > 0 else 0.0
    ),
    "bool": (
        "Answer with a single yes or no!",
        lambda a: "y" in a.lower()
    ),
}

def get_preprompt(key):
    if isinstance(key, str):
        return preprompts[key]
    if isinstance(key, list):
        return generate_onehot(key)
    raise Exception("Unexpected key type in preprompt.")

prompts = [
    # people_and_actions
    ("list[str]", "What are the people wearing in the picture?"),
    ("list[str]", "What is each person doing in the picture?"),
    ("int", "How many people are in the picture?"),
    ("int", "How many people are standing?"),
    ("int", "How many people are sitting?"),
    ("int", "How many people are looking at the camera?"),
    ("int", "How many people are smiling?"),
    ("int", "How many people are using tools?"),
    ("int", "How many people are talking?"),
    ("int", "How many people are handling computer components?"),
    ("int", "How many people appear to be reacting emotionally?"),
    ("int", "How many people are making gestures with their hands?"),
    # objects_and_environment
    ("list[str]", "What types of computer components are in the picture?"),
    ("list[str]", "What tools are being used in the picture?"),
    ("int", "How many computer components are visible in the picture?"),
    ("int", "How many tools are visible in the picture?"),
    ("int", "How many monitors are in the picture?"),
    ("int", "How many keyboards are in the picture?"),
    ("int", "How many mice are in the picture?"),
    ("int", "How many cables are visible in the picture?"),
    ("int", "How many RGB lights are visible in the picture?"),
    ("int", "How many workbenches or tables are visible in the picture?"),
    ("int", "How many cases or chassis are in the picture?"),
    ("int", "How many boxes or packaging materials are visible in the picture?"),
    ("bool", "Is a completed PC visible in the picture?"),
    ("bool", "Is a disassembled PC visible in the picture?"),
    ("bool", "Are there any brand logos visible in the picture?"),
    (["low", "medium", "high"], "How cluttered is the workspace in the picture?"),
    # camera_angles_and_framing
    ("int", "How many faces are clearly visible?"),
    ("int", "How many text elements are in the picture?"),
    ("int", "How many different colors dominate the frame?"),
    ("bool", "Is the image a close-up or wide shot?"),
    ("bool", "Is there text overlay visible in the picture?"),
    (["top-down", "side-view", "front-facing"], "What is the camera angle?"),
    # engagement_and_expression
    ("int", "How many people appear to be laughing?"),
    ("int", "How many people appear to be explaining something?"),
    ("bool", "Are there exaggerated facial expressions in the picture?"),
    ("bool", "Are there any pointing gestures in the frame?"),
    ("bool", "Is anyone making a surprised expression?"),
    ("bool", "Is there a dramatic pose or action in the picture?"),
    # logos_and_branding
    ("list[str]", "What brands are visible in the picture?"),
    ("int", "How many visible brand logos are in the picture?"),
    ("bool", "Is there an LTT logo in the frame?"),
    ("bool", "Is there a sponsor logo visible in the frame?"),
    # textual_elements_and_graphics
    ("int", "How many text elements are present in the frame?"),
    ("int", "How many overlay graphics are present?"),
    ("bool", "Is there an on-screen subtitle or caption in the frame?"),
    ("bool", "Are there any highlighted elements or arrows in the frame?"),
    ("bool", "Does the frame contain a thumbnail-style reaction face?"),
]

# Initialize model and processor
model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2"
).to('cuda')

model = torch.compile(model)

processor = AutoProcessor.from_pretrained(model_id)

def prompt_grouped(image_path, prompts):
    """
    Group prompts by output datatype and submit one LLM call per group.
    """
    # Group prompts by datatype key.
    groups = {}
    for idx, (dtype, question) in enumerate(prompts):
        # For list types, use the tuple of values as key.
        key = tuple(dtype) if isinstance(dtype, list) else dtype
        if key not in groups:
            groups[key] = {"questions": [], "indices": []}
        groups[key]["questions"].append(question)
        groups[key]["indices"].append(idx)

    # Prepare a results list to fill in answers in original prompt order.
    results = [None] * len(prompts)

    # Load image once.
    raw_image = Image.open(image_path)

    # Process each group.
    for key, group in groups.items():
        # Get the preprompt and associated conversion function.
        dtype_key = list(key) if isinstance(key, tuple) else key
        preprompt, conversion_fn = get_preprompt(dtype_key)
        
        # Combine all questions into one prompt.
        numbered_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(group["questions"]))
        full_prompt = f"{preprompt} Answer the following questions in one comma-separated response corresponding to the numbered order:\n{numbered_questions}"
        
        # Create conversation format.
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt},
                {"type": "image"},
            ],
        }]
        
        # Apply chat template.
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process inputs for image and text.
        inputs = processor(images=raw_image, text=prompt_text, return_tensors='pt').to(0, torch.float16)
        
        # Generate output for this group.
        output = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        
        # Decode the output.
        response = processor.decode(output[0][2:], skip_special_tokens=True)
        
        # Process the response using the conversion function.
        # The conversion function is expected to return a list of answers.
        group_answers = conversion_fn(response)
        
        # If the number of answers doesn't match the number of questions,
        # fall back to splitting the response by commas.
        if not isinstance(group_answers, list) or len(group_answers) != len(group["questions"]):
            group_answers = [a.strip() for a in response.split(',') if a.strip() != '']
        
        # Map the parsed answers back to their original positions.
        for local_idx, global_idx in enumerate(group["indices"]):
            results[global_idx] = group_answers[local_idx] if local_idx < len(group_answers) else None

    return results

if __name__ == "__main__":
    image_path = '/mnt-persist/test/1/images/000005_Our_New_4500_Workstation_PCs_for_Editing.jpg'
    
    # Use the PyTorch profiler to capture CPU and CUDA activity.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        grouped_results = prompt_grouped(image_path, prompts)
        prof.step()  # Mark the end of the profiling step

    print("Grouped Results:")
    print(grouped_results)
    print("\nProfiler Summary:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # The following commented-out section can be used to process multiple images.
    # images = os.listdir(path + "/images")
    # images = sorted(images)
    #
    # os.makedirs(path + "/feature_video", exist_ok=True)
    # if debug:
    #     pd.set_option('display.width', os.get_terminal_size().columns)
    #     os.makedirs(path + "/feature_video_debug", exist_ok=True)
    # def ljust(s):
    #     s = s.astype(str).str.strip()
    #     return s.str.ljust(s.str.len().max())
    #
    # for i in range(len(images)):
    #     results = prompt_grouped(path + "/images/" + images[i], prompts)
    #     pickle_save(path + "/feature_video/" + images[i][:-4] + ".pkl", results)
    #     if debug:
    #         prs = np.array([[str(p[0]), str(p[1])] for p in prompts])
    #         prs = np.concatenate([prs, np.array([[str(r) for r in results]]).T], axis=1)
    #         df = pd.DataFrame(prs, columns=['datatype', 'prompt', 'value'])
    #         txt = df.apply(ljust).to_string(index=False, justify='left')
    #         text_save(path + "/feature_video_debug/" + images[i][:-4] + ".txt", txt)
    #     print(str(i+1) + "/" + str(len(images)))
