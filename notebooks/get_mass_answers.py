import os
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ================== CONFIG ==================

# Path to your evaluation set (list of {"question": ..., "answer": ...})
EVAL_FILE = "./../data/eval_qa_pairs.json"

# Where to save the results (will be created/overwritten)
OUTPUT_FILE = "./../eval_outputs/baseline_1b.json"

# Base HF model id (or local path)
BASE_MODEL_ID = "google/gemma-3-1b-it"
# BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# LoRA adapter directory (set to None to use base model only)
# ADAPTER_DIR = "./../outputs/lora-gemma1b"  # or None
ADAPTER_DIR = None

# Generation settings for evaluation (deterministic is nice for comparison)
MAX_NEW_TOKENS = 500
DO_SAMPLE = False       # False = greedy decoding
TEMPERATURE = 0.0       # ignored if DO_SAMPLE=False
TOP_P = 0.9             # ignored if DO_SAMPLE=False


# ================== MODEL LOADING ==================

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_model_and_tokenizer():
    device = get_device()
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_DIR if ADAPTER_DIR and os.path.isdir(ADAPTER_DIR) else BASE_MODEL_ID
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
    )

    if ADAPTER_DIR and os.path.isdir(ADAPTER_DIR):
        print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    else:
        print("No LoRA adapter found/used – using base model only.")
        model = base_model

    model.to(device).eval()
    return model, tokenizer, device


# ================== PROMPT + GENERATION ==================

# def build_prompt(user_question: str) -> str:
#     """
#     IMPORTANT: use the SAME prompt you use in your interactive script,
#     so evaluation is apples-to-apples.
#     """
#     return f"""
# You are a nutrition and recipe assistant.

# First, silently think through the problem in several steps:
# 1) Decide the type of meal that fits the request (cuisine, vegan/vegetarian/non-veg, breakfast/lunch/dinner, etc.).
# 2) Choose a small set of realistic ingredients and assign each a weight in grams or milliliters.
# 3) For each ingredient, estimate:
#    - calories_per_unit (kcal per gram or ml),
#    - total_calories = calories_per_unit * weight,
#    - protein (grams of protein for that ingredient).
# 4) Add up all ingredient calories to get the meal’s total calories.
# 5) Add up all ingredient protein to get the meal’s total protein.
# 6) Check that:
#    - total calories are close to the user’s requested target/range,
#    - total protein is reasonable for the meal or meets the user’s target,
#    - all ingredients respect the user’s dietary constraints.
# 7) Only after checking everything, write the final answer.

# IMPORTANT:
# - Do NOT show your reasoning steps.
# - Only output the final recipe.
# - Use this exact format:

# Recipe name: <name>
# Total calories: <float> kcal
# Total protein: <float> g

# Ingredients:
# - <ingredient>, <weight_g_or_ml> g/ml, <total_calories> kcal, <protein> g
# - <ingredient>, <weight_g_or_ml> g/ml, <total_calories> kcal, <protein> g
# - ...

# User: {user_question}

# Coach:
#     """.strip()

def build_prompt(user_question: str) -> str:
    return f"""
You are a generic recipe assistant.

Answer the user's request with one meal recipe in the following format.

Use exactly this structure:

Recipe name: <name>
Total calories: <number> kcal
Total protein: <number> g

Ingredients:
- <ingredient>, <weight> g/ml, <calories> kcal, <protein> g
- <ingredient>, <weight> g/ml, <calories> kcal, <protein> g
- ...

User: {user_question}

Coach:
""".strip()


@torch.inference_mode()
def generate_answer(model, tokenizer, device: str, user_question: str) -> str:
    prompt = build_prompt(user_question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # strip the prompt tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer.strip()


# ================== EVAL LOOP ==================

def load_eval_data(path: str):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Expecting a list of {"question": ..., "answer": ...}
    if not isinstance(data, list):
        raise ValueError("EVAL_FILE must contain a JSON list of objects.")
    for i, row in enumerate(data):
        if "question" not in row:
            raise ValueError(f"Row {i} missing 'question' key")
        if "answer" not in row:
            raise ValueError(f"Row {i} missing 'answer' key")
    return data


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    eval_data = load_eval_data(EVAL_FILE)
    print(f"Loaded {len(eval_data)} eval examples from {EVAL_FILE}")

    model, tokenizer, device = load_model_and_tokenizer()

    results = []
    for idx, row in enumerate(eval_data, start=1):
        question = row["question"]
        answer = row["answer"]

        print(f"\n[{idx}/{len(eval_data)}] Question:\n{question}\n")

        model_answer = generate_answer(model, tokenizer, device, question)

        print("Model answer (truncated preview):")
        print(model_answer[:300] + ("..." if len(model_answer) > 300 else ""))

        results.append(
            {
                "question": question,
                "answer": answer,
                "model_answer": model_answer,
            }
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(results)} results to {OUTPUT_FILE}")


if __name__ == "__main__":
    # Fix seed if you want strictly reproducible runs
    torch.manual_seed(0)
    main()