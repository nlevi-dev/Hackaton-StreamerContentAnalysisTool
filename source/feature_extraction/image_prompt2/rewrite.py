import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

model_name = "llava-hf/llava-v1.6-mistral-7b-hf"

print(f"Loading processor from: {model_name}")
processor = LlavaNextProcessor.from_pretrained(model_name)

print(f"Loading model from: {model_name}")
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).cuda()
model.eval()

def encode_image_once(model, processor, image: Image.Image):
    print("\n[encode_image_once] ENTERING FUNCTION")
    try:
        print(f"[encode_image_once] Image object type: {type(image)}")
        print(f"[encode_image_once] Image mode: {image.mode}, size: {image.size}")

        # Always supply some text (even if empty) to avoid NoneType errors in processor
        print("[encode_image_once] Calling processor(...) with text='' to ensure we don't have None text.")
        # The result will have "pixel_values" for the image
        inputs = processor(images=image, text="", return_tensors="pt")
        
        print(f"[encode_image_once] After processor call, keys in `inputs`: {list(inputs.keys())}")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"   {k} shape={v.shape}, dtype={v.dtype}, device={v.device}")
        
        # Move to model device
        inputs = inputs.to(model.device)
        pixel_values = inputs["pixel_values"]
        if pixel_values.ndim == 5 and pixel_values.shape[1] == 5:
            pixel_values = pixel_values[:, 0]  # now [1,3,336,336]
        print(f"[encode_image_once] pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}, device: {pixel_values.device}")

        # Some LLaVA variants produce 5D shape [1, 3, H, W, 1].
        if pixel_values.ndim == 5 and pixel_values.shape[-1] == 1:
            pixel_values = pixel_values.squeeze(-1)
            print(f"[encode_image_once] pixel_values squeezed to shape: {pixel_values.shape}")

        with torch.no_grad():
            print("[encode_image_once] Forward pass through vision_tower...")
            vision_outputs = model.vision_tower(pixel_values, output_hidden_states=True)

        all_hidden_states = vision_outputs.hidden_states  # typically a tuple
        print(f"[encode_image_once] vision_outputs.hidden_states is a tuple of length {len(all_hidden_states)}")
        for i, hs in enumerate(all_hidden_states):
            print(f"   hidden_states[{i}]: shape={hs.shape}, dtype={hs.dtype}, device={hs.device}")

        # Usually we take the last one
        patch_embeds = all_hidden_states[-1]
        print(f"[encode_image_once] patch_embeds (last layer) shape: {patch_embeds.shape}")

        # Typically the first token is CLS, so skip it
        patch_embeds = patch_embeds[:, 1:, :]
        print(f"[encode_image_once] patch_embeds after removing CLS shape: {patch_embeds.shape}")

        # Now project to match LLM hidden size
        if hasattr(model, "mm_projector"):
            print("[encode_image_once] Found model.mm_projector; applying it.")
            image_tokens = model.mm_projector(patch_embeds)
        elif hasattr(model, "visual_projection"):
            print("[encode_image_once] Found model.visual_projection; applying it.")
            image_tokens = model.visual_projection(patch_embeds)
        else:
            raise RuntimeError("Cannot find the projection layer in LLaVA model!")

        print(f"[encode_image_once] image_tokens shape: {image_tokens.shape}, dtype={image_tokens.dtype}, device={image_tokens.device}")
        print("[encode_image_once] SUCCESSFUL ENCODE, returning image_tokens.\n")

        return image_tokens

    except Exception as e:
        print("[encode_image_once] An error occurred during image encoding:", e)
        raise


def create_multimodal_inputs(processor, text_prompt: str, num_image_patches: int):
    print("\n[create_multimodal_inputs] ENTERING FUNCTION")
    print(f"[create_multimodal_inputs] text_prompt='{text_prompt}', num_image_patches={num_image_patches}")

    # Minimal conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "fake_image.jpg"},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    print("[create_multimodal_inputs] Created conversation object. Now calling apply_chat_template...")
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    print(f"[create_multimodal_inputs] prompt_text length={len(prompt_text)} chars:\n{prompt_text}\n")

    # NOTE: If the processor does not insert <im_patch> tokens without real images,
    #       you might need to manually do a string replace, or pass `images=...`.
    #       We'll see if it works out-of-the-box for your version.

    print("[create_multimodal_inputs] Now calling processor(...) to get input_ids.")
    inputs = processor(
        images=None,  # We do not want to re-encode real images here
        text=prompt_text,
        return_tensors="pt",
    )
    print(f"[create_multimodal_inputs] Got these keys from processor: {list(inputs.keys())}")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"   {k} shape={v.shape}, dtype={v.dtype}, device={v.device}")
    print("[create_multimodal_inputs] Returning these inputs.\n")
    return inputs


def replace_image_tokens_with_embeds(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_embeds: torch.Tensor,
    processor: LlavaNextProcessor
):
    print("\n[replace_image_tokens_with_embeds] ENTERING FUNCTION")
    print(f"   input_ids shape={input_ids.shape}, dtype={input_ids.dtype}, device={input_ids.device}")
    print(f"   attention_mask shape={attention_mask.shape}, dtype={attention_mask.dtype}, device={attention_mask.device}")
    print(f"   image_embeds shape={image_embeds.shape}, dtype={image_embeds.dtype}, device={image_embeds.device}")

    try:
        im_patch_token_id = processor.tokenizer.convert_tokens_to_ids("<im_patch>")
        print(f"[replace_image_tokens_with_embeds] <im_patch> token_id = {im_patch_token_id}")
    except Exception as e:
        raise RuntimeError("Could not find <im_patch> token_id!") from e

    bsz, seq_len = input_ids.shape
    num_image_patches = image_embeds.shape[1]

    print(f"[replace_image_tokens_with_embeds] bsz={bsz}, seq_len={seq_len}, num_image_patches={num_image_patches}")

    # Find positions of <im_patch>
    patch_positions = (input_ids[0] == im_patch_token_id).nonzero().squeeze(-1)
    print(f"[replace_image_tokens_with_embeds] Found {len(patch_positions)} <im_patch> tokens at positions: {patch_positions.tolist()}")

    if len(patch_positions) != num_image_patches:
        raise ValueError(
            f"Number of <im_patch> tokens in text ({len(patch_positions)}) != shape of image_embeds ({num_image_patches})."
        )

    with torch.no_grad():
        print("[replace_image_tokens_with_embeds] Running model.get_input_embeddings() on input_ids...")
        text_embeds = model.get_input_embeddings()(input_ids)  # shape: [bsz, seq_len, hidden_dim]
        print(f"[replace_image_tokens_with_embeds] text_embeds shape={text_embeds.shape}, dtype={text_embeds.dtype}, device={text_embeds.device}")

        # Replace the patch positions with our cached image tokens
        for i, pos in enumerate(patch_positions):
            text_embeds[0, pos, :] = image_embeds[0, i, :]

    print("[replace_image_tokens_with_embeds] DONE replacing <im_patch> tokens with image_embeds.\n")
    return text_embeds


def generate_with_cached_image(model, processor, text_prompt: str, image_tokens: torch.Tensor):
    print("\n[generate_with_cached_image] ENTERING FUNCTION with text_prompt:", text_prompt)
    try:
        num_patches = image_tokens.size(1)
        print(f"[generate_with_cached_image] num_patches in image_tokens = {num_patches}")

        text_inputs = create_multimodal_inputs(processor, text_prompt, num_patches)
        input_ids = text_inputs["input_ids"].to(model.device)
        attention_mask = text_inputs["attention_mask"].to(model.device)

        print("[generate_with_cached_image] calling replace_image_tokens_with_embeds(...)")
        input_embeds = replace_image_tokens_with_embeds(
            model, input_ids, attention_mask, image_tokens, processor
        )

        print(f"[generate_with_cached_image] input_embeds shape={input_embeds.shape}, dtype={input_embeds.dtype}, device={input_embeds.device}")
        print("[generate_with_cached_image] Generating...")
        with torch.no_grad():
            generated_ids = model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=80,
                do_sample=False,
                temperature=0.3
            )

        output_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        print("[generate_with_cached_image] Generated text:", output_text)
        return output_text

    except Exception as e:
        print("[generate_with_cached_image] ERROR:", e)
        raise


if __name__ == "__main__":
    print("\n[MAIN] Loading the test image...\n")
    image_path = "/mnt-persist/test/1/images/000005_Our_New_4500_Workstation_PCs_for_Editing.jpg"
    image = Image.open(image_path).convert("RGB")

    print("[MAIN] Now calling encode_image_once...")
    cached_image_tokens = encode_image_once(model, processor, image)

    # If encoding worked, proceed
    questions = [
        ("list[str]", "What are the people wearing in the picture?"),
        ("list[str]", "What is each person doing in the picture?"),
        ("int", "How many people are in the picture?"),
        ("int", "How many people are standing?"),
    ]

    print("\n[MAIN] Reusing cached_image_tokens for multiple questions...\n")
    for key, prompt in questions:
        # Combine your "prompt" with any prefix you like based on key
        # For now, we just pass prompt. Insert whatever "Answer with an integer" etc. if needed:
        question_text = f"{prompt}"
        print("-----------------------------------------------------")
        print("Question:", question_text)
        try:
            answer = generate_with_cached_image(model, processor, question_text, cached_image_tokens)
            print("Answer:", answer, "\n")
        except Exception as e:
            print("Error during generation:", e)
            break
