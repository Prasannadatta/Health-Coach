import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "google/gemma-3-1b-it"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "lora-gemma1b-healthcoach")
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
    return f"""
    You are a nutrition and recipe assistant.

    First, silently think through the problem in several steps:
    1) Decide the type of meal that fits the request (cuisine, vegan/vegetarian/non-veg, breakfast/lunch/dinner, etc.).
    2) Choose a small set of realistic ingredients and assign each a weight in grams or milliliters.
    3) For each ingredient, estimate:
    - calories_per_unit (kcal per gram or ml),
    - total_calories = calories_per_unit * weight,
    - protein (grams of protein for that ingredient).
    4) Add up all ingredient calories to get the meal total calories.
    5) Add up all ingredient protein to get the meal total protein.
    6) Check that:
    - total calories are close to the requested target/range,
    - total protein is reasonable for the meal or meets the target,
    - all ingredients respect the dietary constraints given by the user.
    7) Only after checking everything, write the final answer.

    IMPORTANT:
    - Do NOT show your reasoning steps.
    - Only output the final recipe.
    - Use this exact format:

    Recipe name: <name>
    Total calories: <float> kcal
    Total protein: <float> g

    Ingredients:
    - <ingredient>, <weight_g_or_ml> g/ml, <total_calories> kcal, <protein> g
    - <ingredient>, <weight_g_or_ml> g/ml, <total_calories> kcal, <protein> g
    - ...

    Follow the style of these examples:

    Example 1
    User: Give me an Indian vegetarian dinner under 600 calories with at least 25 g protein.

    Coach:
    Recipe name: Lentil, Spinach & Rice Bowl with Yogurt
    Total calories: 517 kcal
    Total protein: 30 g

    Ingredients:
    - Cooked red lentils, 150 g, 174 kcal, 13.5 g
    - Fresh spinach, 80 g, 18 kcal, 2.3 g
    - Cooked brown rice, 160 g, 178 kcal, 4.2 g
    - Plain low-fat yogurt, 100 g, 59 kcal, 10 g
    - Olive oil, 10 g, 88 kcal, 0.0 g


    Example 2
    User: Give me a vegan breakfast around 400 calories with decent protein.

    Coach:
    Recipe name: Peanut Butter Banana Overnight Oats
    Total calories: 433 kcal
    Total protein: 18.5 g

    Ingredients:
    - Rolled oats, 50 g, 195 kcal, 8.5 g
    - Unsweetened soy milk, 200 ml, 108 kcal, 6.6 g
    - Banana, 80 g, 71 kcal, 0.9 g
    - Peanut butter, 10 g, 59 kcal, 2.5 g


    Now answer the next request in the SAME FORMAT.

    User: {user_question}

    Coach:
    """.strip()


def generate_answer(model, tokenizer, user_question: str) -> str:
    prompt = build_prompt(user_question)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,      # length of answer
            temperature=0.1,         # creativity
            do_sample=False,          # sampling instead of greedy
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
        print(reply)


if __name__ == "__main__":
    main()
