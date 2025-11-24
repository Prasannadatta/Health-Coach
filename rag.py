import pandas as pd
import json
from pathlib import Path
import numpy as np
import json

import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import PeftModel

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME = "google/gemma-3-270m"
DOCS_PATH = "docs.jsonl"
FAISS_PATH = "faiss.index"

ADAPTER_DIR   = "Health-Coach/outputs/lora-llama3.2-1b-instruct_1epochs"



device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# device = "cpu"

# load docs + index
docs = [json.loads(l) for l in open(DOCS_PATH, "r", encoding="utf-8")]
index = faiss.read_index(FAISS_PATH)

embedder = SentenceTransformer(EMBED_MODEL)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
    )

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.to(device).eval()

def retrieve(query, k=6):
    q = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    results = []
    for rank, i in enumerate(I[0]):
        if i < len(docs):
            doc = docs[i]
            results.append((doc["instruction"] + "\n" + doc["output"], float(D[0][rank])))
    return results

# if __name__ == "__main__":

#     query = "I need a meal with 1550 calories with 95g protein."
#     calorie_target = 1550
#     protein_target = 95

#     # print("Retrieving relevant documents...")

#     # retrieved = retrieve(query, k=12)
#     # context = "\n\n---\n".join([r[0] for r in retrieved])

#     retrieved = retrieve(query, k=4)  # fewer docs

#     # take only the first ~200 words from each doc to keep prompt small
#     texts = [r[0] for r in retrieved]
#     trimmed = [" ".join(t.split()[:200]) for t in texts]

#     context = "\n\n---\n".join(trimmed)

#     print("Generating response...") 

#     prompt = '''
#         You are a nutrition and recipe assistant. 
#         Format your answer exactly as valid JSON with this schema:
#         {{"meal_plan": {{"breakfast": [], "lunch": [], "dinner": []}}, "totals": {{"calories": 0, "protein": 0}}}}

#         The below is an example question-answer pair of a meal plan for someone needing 1550 calories and 95g protein:

#             QUESTION: I have prediabetes and need 1550 calories with 95g protein to manage blood sugar.

#             ANSWER: {{"meal_plan": {{"breakfast": [{{"item": "Scrambled Eggs (4 oz)", "calories": 141, "protein": 11}}, {{"item": "Turkey Breast (3 oz)", "calories": 76, "protein": 14}}, {{"item": "Egg Whites (4 oz)", "calories": 53, "protein": 11}}, {{"item": "Hard Boiled Eggs (1 each)", "calories": 76, "protein": 7}}, {{"item": "Baby Spinach (4 oz)", "calories": 7, "protein": 1}}, {{"item": "Broccoli (2 oz)", "calories": 19, "protein": 2}}, {{"item": "Sliced Mushrooms (1 oz)", "calories": 6, "protein": 1}}, {{"item": "Diced Tomatoes (2 oz)", "calories": 15, "protein": 1}}, {{"item": "Strawberries (0.5 oz)", "calories": 4, "protein": 0}}, {{"item": "Blueberries (2 oz)", "calories": 32, "protein": 0}}], "lunch": [{{"item": "Liquid Eggs (3 oz)", "calories": 126, "protein": 10}}, {{"item": "Turkey Breast (3 oz)", "calories": 76, "protein": 14}}, {{"item": "Shredded Mozzarella Cheese (1 oz)", "calories": 91, "protein": 6}}, {{"item": "Boiled Ham Sliced (1 oz)", "calories": 30, "protein": 4}}, {{"item": "Savory Plant-Based Breakfast Patties (1 each)", "calories": 70, "protein": 6}}, {{"item": "Vegan Egg (1 oz)", "calories": 44, "protein": 3}}, {{"item": "Diced Green Bell Peppers (1 oz)", "calories": 6, "protein": 0}}, {{"item": "Diced Red Bell Peppers (1 oz)", "calories": 9, "protein": 0}}, {{"item": "Chopped Onions (1 oz)", "calories": 12, "protein": 0}}, {{"item": "Red Seedless Grapes (1 cup)", "calories": 156, "protein": 2}}], "dinner": [{{"item": "Belgian Waffles (1 each)", "calories": 129, "protein": 3}}, {{"item": "Pork Sausage Crumbles (2 oz)", "calories": 190, "protein": 7}}, {{"item": "Shredded Cheddar Cheese (1 oz)", "calories": 114, "protein": 7}}, {{"item": "Pork Roll (2 oz)", "calories": 182, "protein": 9}}, {{"item": "Oatmeal (6 oz)", "calories": 146, "protein": 5}}, {{"item": "Watermelon (1 wedge)", "calories": 108, "protein": 2}}, {{"item": "Cantaloupe Melon (1 slice)", "calories": 39, "protein": 1}}, {{"item": "Honeydew Melon (1 wedge)", "calories": 51, "protein": 1}}]}}, "totals": {{"calories": 1550, "protein": 95}}

#         Use the context below (which contains nutrition facts and recipe info).

#         Context:
#         {context}

#         Can you create a similar meal plan for the question below?

#         QUESTION: I have diabetes and need to watch my carbs. Can you create a 1400-calorie, low-carb meal plan with 100g protein?

#         Return ONLY the answer in valid JSON.
#     '''.format(context=context)

#     # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,  # important
#         max_length= model.config.max_position_embeddings - 256  # leave room for 256 new tokens
#     ).to(model.device)

#     print("model max context:", model.config.max_position_embeddings)
#     print("input token length:", inputs["input_ids"].shape[1])

#     output_ids = model.generate(
#         **inputs,
#         max_new_tokens=512,
#         top_k=50,
#         # temperature=0.7,
#         # top_p=0.9,
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#     )

#     print("output token length:", output_ids.shape[1])

#     # prompt_len = inputs["input_ids"].shape[1]

#     # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     # gen_ids = output_ids[0][prompt_len:]

#     # result = tokenizer.decode(gen_ids, skip_special_tokens=True)
#     # print("Model answer only:\n", result)
#     # Decode EVERYTHING (prompt + generation)
#     full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     print("FULL TEXT (for debugging):\n", repr(full_text))

#     # Strip the prompt text once from the front to get ONLY the answer
#     answer_only = full_text.replace(prompt, "", 1).strip()
#     print("Model answer only:\n", answer_only)

if __name__ == "__main__":

    query = "I need a meal with 1550 calories with 95g protein."
    calorie_target = 1550
    protein_target = 95

    print("Retrieving relevant documents...")

    # 🔹 smaller RAG context
    retrieved = retrieve(query, k=4)
    texts = [r[0] for r in retrieved]
    trimmed = [" ".join(t.split()[:200]) for t in texts]  # first ~200 words/doc
    context = "\n\n---\n".join(trimmed)

    print("Generating response...")

    # prompt = f"""
    # You are a nutrition and recipe assistant.

    # Use the nutrition context below, create a 1400-calorie, low-carb meal plan with 100g protein
    # for someone with diabetes who needs to watch their carbs. Don't deviate from these requirements.

    # You MUST answer as valid JSON with exactly this schema:
    # {{"meal_plan": {{"breakfast": [], "lunch": [], "dinner": []}},
    # "totals": {{"calories": 1550, "protein": 95}}}}

    # Context:
    # {context}

    # """

    prompt = f'''
    You are a nutrition and recipe assistant.
    Your job is to pick ONE realistic meal recipe that matches the user's request
    (cuisine, calories, protein target, diet type like vegan/vegetarian/non-veg,
    meal type like breakfast/lunch/dinner, etc.).

    You MUST respond ONLY with a single Python dictionary literal (no extra text)
    with this exact structure:
    {{
    'recipe_name': str,                 # name of the recipe
    'calories': float,                  # total kcal per serving
    'proteins': float,                  # grams of protein per serving
    'ingredients': [
        {{
            'ingredient': str,            # ingredient name
            'weight_g_or_ml': float,      # amount per serving in g or ml
            'calories_per_unit': float,   # kcal per g or ml
            'total_calories': float,      # kcal from this ingredient per serving
            'protein': float              # grams of protein from this ingredient per serving
        }},
        ...
    ]
    }}

    Requirements:
    - Make calories and protein values realistic and consistent.
    - The sum of ingredient 'total_calories' should be close to 'calories'.
    - The sum of ingredient 'protein' should be close to 'proteins'.
    - The recipe MUST respect the user's calorie / protein / cuisine / diet constraints.
    - Do NOT include explanations, comments, or any text outside the dictionary.

    Give me a vegan meal under 700 calories with at least 50g protein for lunch.

    '''


    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096   # leave room
    ).to(model.device)

    print("model max context:", model.config.max_position_embeddings)
    print("input token length:", inputs["input_ids"].shape[1])

    output_ids = model.generate(
        **inputs,
        max_new_tokens=2560,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    print("output token length:", output_ids.shape[1])

    
    # 1) Length of the prompt in tokens
    prompt_len = inputs["input_ids"].shape[1]

    # 2) Slice off the prompt tokens → only generated tokens remain
    gen_ids = output_ids[0][prompt_len:]

    # 3) Decode ONLY the generated tokens
    answer_only = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print("Model answer only:\n", repr(answer_only))

    # (optional) still see what the model did overall
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("FULL TEXT (for debugging):\n", repr(full_text))

    # (optional) last few raw tokens
    tail_raw = tokenizer.decode(output_ids[0][-20:], skip_special_tokens=False)
    print("Last tokens (raw):\n", repr(tail_raw))

#   You are a nutrition and recipe assistant. 
# Format your answer exactly as valid JSON with this schema:
# {{"meal_plan": {{"breakfast": [], "lunch": [], "dinner": []}}, "totals": {{"calories": 0, "protein": 0}}}}

# The below is an example question-answer pair of a meal plan for someone needing 1550 calories and 95g protein:

# Q: I have prediabetes and need 1550 calories with 95g protein to manage blood sugar.

# A: {"meal_plan": {"breakfast": [{"item": "Scrambled Eggs (4 oz)", "calories": 141, "protein": 11}, {"item": "Turkey Breast (3 oz)", "calories": 76, "protein": 14}, {"item": "Egg Whites (4 oz)", "calories": 53, "protein": 11}, {"item": "Hard Boiled Eggs (1 each)", "calories": 76, "protein": 7}, {"item": "Baby Spinach (4 oz)", "calories": 7, "protein": 1}, {"item": "Broccoli (2 oz)", "calories": 19, "protein": 2}, {"item": "Sliced Mushrooms (1 oz)", "calories": 6, "protein": 1}, {"item": "Diced Tomatoes (2 oz)", "calories": 15, "protein": 1}, {"item": "Strawberries (0.5 oz)", "calories": 4, "protein": 0}, {"item": "Blueberries (2 oz)", "calories": 32, "protein": 0}], "lunch": [{"item": "Liquid Eggs (3 oz)", "calories": 126, "protein": 10}, {"item": "Turkey Breast (3 oz)", "calories": 76, "protein": 14}, {"item": "Shredded Mozzarella Cheese (1 oz)", "calories": 91, "protein": 6}, {"item": "Boiled Ham Sliced (1 oz)", "calories": 30, "protein": 4}, {"item": "Savory Plant-Based Breakfast Patties (1 each)", "calories": 70, "protein": 6}, {"item": "Vegan Egg (1 oz)", "calories": 44, "protein": 3}, {"item": "Diced Green Bell Peppers (1 oz)", "calories": 6, "protein": 0}, {"item": "Diced Red Bell Peppers (1 oz)", "calories": 9, "protein": 0}, {"item": "Chopped Onions (1 oz)", "calories": 12, "protein": 0}, {"item": "Red Seedless Grapes (1 cup)", "calories": 156, "protein": 2}], "dinner": [{"item": "Belgian Waffles (1 each)", "calories": 129, "protein": 3}, {"item": "Pork Sausage Crumbles (2 oz)", "calories": 190, "protein": 7}, {"item": "Shredded Cheddar Cheese (1 oz)", "calories": 114, "protein": 7}, {"item": "Pork Roll (2 oz)", "calories": 182, "protein": 9}, {"item": "Oatmeal (6 oz)", "calories": 146, "protein": 5}, {"item": "Watermelon (1 wedge)", "calories": 108, "protein": 2}, {"item": "Cantaloupe Melon (1 slice)", "calories": 39, "protein": 1}, {"item": "Honeydew Melon (1 wedge)", "calories": 51, "protein": 1}]}, "totals": {"calories": 1550, "protein": 95}}

# Use the context below (which contains nutrition facts and recipe info).

# Context:
# {context}

# Can you create a similar meal plan for the question below?

# Q: I have diabetes and need to watch my carbs. Can you create a 1400-calorie, low-carb meal plan with 100g protein?

# Return ONLY valid JSON.


# import pandas as pd
# import json
# from pathlib import Path
# import numpy as np
# import re
# import json


# from transformers import pipeline
# import torch
# import transformers
# import faiss
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import load_dataset
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


# class RAGIndex:
#     def __init__(self, embed_model="sentence-transformers/all-MiniLM-L6-v2"):
#         self.embedder = SentenceTransformer(embed_model)
#         self.index = None
#         self.docs = []

#     def build_index(self, docs):
#         self.docs = docs
#         texts = [(d["instruction"] + "\n" + d["output"]) for d in docs]

#         print("Encoding docs...")
#         emb = self.embedder.encode(texts, convert_to_numpy=True, batch_size=64).astype("float32")

#         faiss.normalize_L2(emb)

#         dim = emb.shape[1]
#         index = faiss.IndexFlatIP(dim)
#         index.add(emb)

#         self.index = index
#         print(f"FAISS index built with {len(docs)} documents.")

#     def retrieve(self, query, k=3):
#         q = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
#         faiss.normalize_L2(q)

#         scores, idxs = self.index.search(q, k)
#         return [(self.docs[i]["instruction"] + "\n" + self.docs[i]["output"], float(scores[0][n])) for n, i in enumerate(idxs[0])]
    
#     def save(self, index_path="faiss.index", docs_path="docs.jsonl"):
#         faiss.write_index(self.index, index_path)
#         with open(docs_path, "w", encoding="utf-8") as f:
#             for d in self.docs:
#                 f.write(json.dumps(d, ensure_ascii=False) + "\n")


#     def load(self, index_path="faiss.index", docs_path="docs.jsonl"):
#         self.index = faiss.read_index(index_path)
#         self.docs = [json.loads(line) for line in open(docs_path, "r", encoding="utf-8")]



# class RAGModel:
#     def __init__(self, model_name="meta-llama/Llama-3.2-1B"):
#         print("Loading model:", model_name)

#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#             device_map="auto"
#         )

#     def answer(self, query, context):
#         prompt = (
#             "You are a nutrition/fitness assistant. Use ONLY the context below.\n\n"
#             f"Context:\n{context}\n\n"
#             f"Question: {query}\nAnswer:"
#         )

#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

#         output = self.model.generate(
#             **inputs,
#             max_new_tokens=700,
#             temperature=0.7,
#             top_p=0.9
#         )

#         result = self.tokenizer.decode(output[0], skip_special_tokens=True)

#         # Remove the prompt part → only answer
#         if "Answer:" in result:
#             result = result.split("Answer:")[-1].strip()

#         return result


# docs = [json.loads(line) for line in open("instruction_data.jsonl", "r", encoding="utf-8")]

# index = RAGIndex()
# index.build_index(docs)
# index.save()
