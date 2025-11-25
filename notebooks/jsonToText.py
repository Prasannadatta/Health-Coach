import json
from pathlib import Path

def load_examples(path: Path):
    """Load a JSON file as a list of {question, answer} dicts."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # data can be a list of objects or a dict containing such a list
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        # try to find the first list value that looks like examples
        rows = None
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                rows = v
                break
        if rows is None:
            raise ValueError(f"Don't know how to interpret JSON in {path}")
    else:
        raise ValueError(f"Unexpected top-level type in {path}: {type(data)}")

    # sanity check
    cleaned = []
    for r in rows:
        if "question" in r and "answer" in r:
            cleaned.append(r)
        else:
            # skip weird rows
            continue
    return cleaned


def format_answer_as_text(answer_obj: dict) -> str:
    """
    Convert the nested 'answer' object into a text block like:

    Recipe name: ...
    Total calories: ... kcal
    Total protein: ... g

    Ingredients:
    - name, weight g/ml, total_calories kcal, protein g

    Also:
    - drop ingredients where BOTH total_calories == 0 and protein == 0
    - recompute total calories & protein from the remaining ingredients
    """
    name = answer_obj.get("recipe_name", "Unknown recipe")

    raw_ingredients = answer_obj.get("ingredients", []) or []

    # ---- Filter ingredients: drop ones that contribute nothing ----
    filtered_ingredients = []
    for ing in raw_ingredients:
        kcal = float(ing.get("total_calories", 0) or 0.0)
        prot = float(ing.get("protein", 0) or 0.0)

        # skip if both are zero
        if kcal == 0.0 and prot == 0.0:
            continue

        filtered_ingredients.append(ing)

    # ---- Recompute totals from filtered ingredients ----
    if filtered_ingredients:
        total_cal = sum(float(ing.get("total_calories", 0) or 0.0)
                        for ing in filtered_ingredients)
        total_prot = sum(float(ing.get("protein", 0) or 0.0)
                         for ing in filtered_ingredients)
    else:
        # fallback to top-level if everything got filtered out
        total_cal = float(answer_obj.get("calories", 0) or 0.0)
        total_prot = float(
            answer_obj.get("proteins", answer_obj.get("protein", 0)) or 0.0
        )

    lines = []
    lines.append(f"Recipe name: {name}")
    lines.append(f"Total calories: {total_cal:.2f} kcal")
    lines.append(f"Total protein: {total_prot:.2f} g")
    lines.append("")  # blank line
    lines.append("Ingredients:")

    for ing in filtered_ingredients:
        ing_name = ing.get("ingredient", "Unknown ingredient")
        weight = ing.get("weight_g_or_ml", 0)
        kcal = float(ing.get("total_calories", 0) or 0.0)
        prot = float(ing.get("protein", 0) or 0.0)

        lines.append(f"- {ing_name}, {weight} g, {kcal:.2f} kcal, {prot:.2f} g")

    return "\n".join(lines)


def convert_file(input_path: str, output_path: str):
    in_path = Path(input_path)
    out_path = Path(output_path)

    rows = load_examples(in_path)

    converted = []
    for r in rows:
        q = r["question"]
        ans_obj = r["answer"]

        ans_text = format_answer_as_text(ans_obj)

        converted.append({
            "question": q,
            "answer": ans_text
        })

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(converted)} examples to {out_path}")


if __name__ == "__main__":
    input_files = [
        "data/qa_pairs_1.json",
        "data/qa_pairs_2.json",
        "data/qa_pairs_h.json",
        "data/qa_pairs_n.json",
        "data/qa_pairs.json",
        "data/qa_2.json"
    ]

    all_converted = []

    for in_path in input_files:
        rows = load_examples(Path(in_path))
        for r in rows:
            q = r["question"]
            ans_obj = r["answer"]
            ans_text = format_answer_as_text(ans_obj)
            all_converted.append({"question": q, "answer": ans_text})

    out_path = Path("data/qa_pairs_text_merged.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_converted, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(all_converted)} examples to {out_path}")