import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "google/gemma-3-270m"
ADAPTER_DIR = "lora-gemma270m-base-adapter"  # where trainer saved LoRA
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model_and_tokenizer():
    # Load tokenizer from adapter dir (it’s the same as base, but with pad_token set)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # dtype similar to training
    dtype = torch.bfloat16 if DEVICE == "mps" else torch.float32

    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=dtype,
    )

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.to(DEVICE).eval()

    return model, tokenizer


def build_prompt(user_question: str) -> str:
    return (
        "You are a supportive, non-judgmental health and wellness coach. "
        "You give realistic, safe advice about nutrition, exercise, and daily habits.\n\n"
        f"User: {user_question}\n\n"
        "Coach:"
    )


def generate_answer(model, tokenizer, user_question: str) -> str:
    prompt = build_prompt(user_question)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,      # length of answer
            temperature=0.7,         # creativity
            top_p=0.9,
            do_sample=True,          # sampling instead of greedy
        )

    # strip the prompt tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer.strip()


def main():
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    print("Ready! 🔥\n")

    # simple REPL
    while True:
        try:
            user_q = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_q.strip():
            continue

        reply = generate_answer(model, tokenizer, user_q)
        print("\nCoach:", reply)


if __name__ == "__main__":
    main()
