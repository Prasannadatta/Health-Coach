# build_docs.py
import pandas as pd
import json
from pathlib import Path

recipe_path = Path("Health-Coach/nutrition_recipe_processed.csv/nutrition_recipe_processed.csv")
nutrition_path = Path("Data/nutrition.xlsx")

recipe_df = pd.read_csv(recipe_path)
nutrition_df = pd.read_excel(nutrition_path)

recipe_df = recipe_df.fillna("").rename(columns=lambda c: str(c).strip())
nutrition_df = nutrition_df.fillna("").rename(columns=lambda c: str(c).strip())

docs = []

# recipe -> doc
for i, r in recipe_df.iterrows():
    # if i == 40000:
    #     break
    text = " | ".join([f"{col}: {r[col]}" for col in recipe_df.columns if r[col] not in (None, "")])
    docs.append({"id": f"recipe_{i}", "type": "recipe", "instruction": f"Nutrition for {r.get('recipe_name','recipe')}", "input": "", "output": text})

# nutrition -> doc
for i, r in nutrition_df.iterrows():
    text = " | ".join([f"{col}: {r[col]}" for col in nutrition_df.columns if r[col] not in (None, "")])
    docs.append({"id": f"nut_{i}", "type": "ingredient", "instruction": f"What are the nutrition facts of {r.get('name','food')}", "input": "", "output": text})

out_path = Path("combined_docs.jsonl")
with out_path.open("w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print("Wrote", len(docs), "documents ->", out_path)





# import pandas as pd
# import json
# from pathlib import Path
# import numpy as np
# import re
# import json

# def clean_nutrition(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
#     df = df.drop_duplicates().fillna("")
#     if "calories" in df.columns:
#         df["calories"] = pd.to_numeric(df["calories"], errors="coerce")
#     return df

# def clean_recipe_df(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
#     df = df.drop_duplicates().fillna("")
#     # ensure numeric columns where applicable
#     for col in ["calorie_count", "serving_size_gms"]:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce")
#     return df

# def nutrition_to_docs(df):
#     docs = []
#     for _, row in df.iterrows():
#         title = row.get('food', '') or row.get('name', '')
#         lines = []
#         metadata_lines = [
#             f"{col}: {val}"
#             for col, val in row.items()
#             if col not in ("food", "name") and val not in ("", None)
#         ]
#         text = f"{title}\n" + "\n".join(metadata_lines)
#         docs.append({'id': title+str(_), 'text': text})
#     return docs

# def recipe_row_to_doc(row: pd.Series) -> dict:
#     title = row.get("recipe_name", "") or row.get("name", "")
#     # pick a set of useful nutrition/metadata columns to include
#     metadata_cols = [c for c in row.index if c not in ("recipe_name", "ingredients", "instructions")]
#     metadata_lines = []
#     for c in metadata_cols:
#         v = row[c]
#         if v is None or v == "":
#             continue
#         metadata_lines.append(f"{c}: {v}")
#     ing = row.get("ingredients", "")
#     text = f"Recipe: {title}\nIngredients: {ing}\n" + "\n".join(metadata_lines)
#     return {"id": f"recipe_{title}", "instruction": f"What are the nutrition & recipe details for {title}?", "input": "", "output": text}


# def make_instruction_examples(nutrition_df, out_path='instruction_data.jsonl'):
#     examples = []

#     sample_size = len(nutrition_df)
#     sample_nutrition = nutrition_df.sample(sample_size, random_state=42)

#     for _, r in sample_nutrition.iterrows():
#         food = r.get('food', r.get('name','Unknown food'))
#         calories = r.get('calories','unknown')
#         instr = f"What are the main nutrition facts of {food}?"
#         resp = f"{food} — calories: {calories}. " + "; ".join(f"{k}: {v}" for k,v in r.items() if k not in ('food','name') and v not in (None,''))
#         examples.append({"instruction": instr, "input": "", "output": resp})
    
#     # for _, r in exercise_df.sample(min(1000, len(exercise_df))).iterrows():
#     #     ex = r.get('exercise', r.get('name','exercise'))
#     #     instr = f"How many calories does {ex} burn per minute for a 70 kg person?"
#     #     # Placeholder -- compute if MET data exists
#     #     met = r.get('met', None)
#     #     if met:
#     #         resp = f"A rough estimate for a 70 kg person: calories/min = {met*3.5*70/200:.2f} kcal per minute (using MET formula)."
#     #     else:
#     #         resp = "MET value not provided; can't compute without MET or intensity."
#     #     examples.append({"instruction": instr, "input": "", "output": resp})
#     # Write jsonl
#     with open(out_path, 'w', encoding='utf-8') as f:
#         for ex in examples:
#             f.write(json.dumps(ex, ensure_ascii=False) + '\n')
#     print(f"Wrote {len(examples)} examples to {out_path}")

# def load_recipe_table(path: str):
#     # try CSV then Excel
#     p = Path(path)
#     if p.suffix.lower() in (".csv",):
#         df = pd.read_csv(path)
#     else:
#         # try reading xlsx/csv; if image provided, user needs to supply real CSV/XLSX
#         try:
#             df = pd.read_csv(path)
#         except Exception:
#             df = pd.read_excel(path, engine="openpyxl")
#     return df

# nutrition = pd.read_excel('Data/nutrition.xlsx', index_col=0)
# # exercise = pd.read_csv('Data/megaGymDataset.csv')

# nutrition = clean_nutrition(nutrition)
# nutrition_docs = nutrition_to_docs(nutrition)
# # exercise = exercise.fillna('')

# pd.DataFrame(nutrition_docs).to_json('nutrition_docs.jsonl', orient='records', lines=True)
# make_instruction_examples(nutrition, out_path='instruction_data.jsonl')