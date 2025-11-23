import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

# 1. Choose which model to use (you already accepted the license on HF)
# model_id = "google/gemma-3-1b-it"
model_id = "google/gemma-3-270m-it"

# 2. Pick a device: Apple GPU (mps), Nvidia GPU (cuda), or CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

# 3. Load tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Gemma3ForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16 if device != "cpu" else torch.float32,
).to(device).eval()

# 4. Write your zero-shot prompt (no examples, just instructions + question)
prompt = (
    """give me a dinner with 1200 calories"""
)

# 5. Turn the prompt into model input tensors
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 6. Ask the model to generate a continuation of the prompt
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        do_sample = True,
        temperature = 0.5,
        top_p = 0.9,
        repetition_penalty = 1.05,
        max_new_tokens = 800,
    )

# 7. Decode ONLY the new tokens (skip the prompt part)
generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("\nModel answer:\n")
print(answer)