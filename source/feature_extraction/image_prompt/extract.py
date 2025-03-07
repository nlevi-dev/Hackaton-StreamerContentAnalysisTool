import os
while 'source' not in os.listdir():
    os.chdir('..')

import re
import jaro
from transformers import pipeline

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
    prepromt = "Answer from the following list with only using a word from it ["+",".join(items)+"]!"
    closest = lambda a:get_closest_idx(items,a)
    return (prepromt, closest)

preprompts = {
    "list[str]":("Answer with a short, comma separated list!",lambda a:[re.sub("[\\s]+","_",b.strip().lower()) for b in a.split(',') if b.strip() != '']),
    "list[int]":("Answer with a short, comma separated list!",lambda a:[int(re.sub("[^\\d]","",b)) for b in a.split(',') if len(b.strip())>0]),
    "list[float]":("Answer with a short, comma separated list!",lambda a:[float(re.sub("[^\\d\\.]","",b)) for b in a.split(',') if b.strip() != '']),
    "int":("Answer with a single integer number!",lambda a:int(re.sub("[^\\d]","",a)) if len(re.sub("[^\\d]","",a))>0 else 0),
    "float":("Answer with a single float number!",lambda a:float(re.sub("[^\\d\\.]","",a)) if len(re.sub("[^\\d\\.]","",a))>0 else 0.0),
    "bool":("Answer with a single yes or no!",lambda a:"y" in a),
}

def get_preprompt(key):
    if isinstance(key, str):
        return preprompts[key]
    if isinstance(key, list):
        return generate_onehot(key)
    raise Exception(":))))")

prompts = [
    #people_and_actions
    ("list[str]","What are the people wearing in the picture?"),
    ("list[str]","What is each person doing in the picture?"),
    ("int","How many people are in the picture?"),
    ("int","How many people are standing?"),
    ("int","How many people are sitting?"),
    ("int","How many people are looking at the camera?"),
    ("int","How many people are smiling?"),
    ("int","How many people are using tools?"),
    ("int","How many people are talking?"),
    ("int","How many people are handling computer components?"),
    ("int","How many people appear to be reacting emotionally?"),
    ("int","How many people are making gestures with their hands?"),
    #objects_and_environment
    ("list[str]","What types of computer components are in the picture?"),
    ("list[str]","What tools are being used in the picture?"),
    ("int","How many computer components are visible in the picture?"),
    ("int","How many tools are visible in the picture?"),
    ("int","How many monitors are in the picture?"),
    ("int","How many keyboards are in the picture?"),
    ("int","How many mice are in the picture?"),
    ("int","How many cables are visible in the picture?"),
    ("int","How many RGB lights are visible in the picture?"),
    ("int","How many workbenches or tables are visible in the picture?"),
    ("int","How many cases or chassis are in the picture?"),
    ("int","How many boxes or packaging materials are visible in the picture?"),
    ("bool","Is a completed PC visible in the picture?"),
    ("bool","Is a disassembled PC visible in the picture?"),
    ("bool","Are there any brand logos visible in the picture?"),
    (["low","medium","high"],"How cluttered is the workspace in the picture?"),
    #camera_angles_and_framing
    ("int","How many faces are clearly visible?"),
    ("int","How many text elements are in the picture?"),
    ("int","How many different colors dominate the frame?"),
    ("bool","Is the image a close-up or wide shot?"),
    ("bool","Is there text overlay visible in the picture?"),
    (["top-down","side-view","front-facing"],"What is the camera angle?"),
    #engagement_and_expression
    ("int","How many people appear to be laughing?"),
    ("int","How many people appear to be explaining something?"),
    ("bool","Are there exaggerated facial expressions in the picture?"),
    ("bool","Are there any pointing gestures in the frame?"),
    ("bool","Is anyone making a surprised expression?"),
    ("bool","Is there a dramatic pose or action in the picture?"),
    #logos_and_branding
    ("list[str]","What brands are visible in the picture?"),
    ("int","How many visible brand logos are in the picture?"),
    ("bool","Is there an LTT logo in the frame?"),
    ("bool","Is there a sponsor logo visible in the frame?"),
    #textual_elements_and_graphics
    ("int","How many text elements are present in the frame?"),
    ("int","How many overlay graphics are present?"),
    ("bool","Is there an on-screen subtitle or caption in the frame?"),
    ("bool","Are there any highlighted elements or arrows in the frame?"),
    ("bool","Does the frame contain a thumbnail-style reaction face?"),
]

pipe = pipeline("image-text-to-text", model="llava-hf/llava-v1.6-mistral-7b-hf")

def prompt(image, prompts):
    results = []
    for prompt in prompts:
        pre = get_preprompt(prompt[0])
        txt = pre[0]+' '+prompt[1]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                    {"type": "text", "text": txt},
                ],
            },
        ]
        print(prompt[1])
        result = pipe(text=messages, max_new_tokens=20)[0]['generated_text'][-1]['content']
        result = pre[1](result)
        print(result)
        results.append(result)
    return results

prompt('test/1/images/000585_Our_New_4500_Workstation_PCs_for_Editing.jpg', prompts)