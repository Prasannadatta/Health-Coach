import re
import json
from pathlib import Path


def parse_recipe_text(text: str) -> dict:
    # Require at least one digit, optional decimal part
    float_pattern = r"(\d+(?:\.\d+)?)"

    # --- Header fields ---
    name_match = re.search(r"Recipe name:\s*(.+)", text)
    calories_match = re.search(rf"Total calories:\s*{float_pattern}", text)
    protein_match = re.search(rf"Total protein:\s*{float_pattern}", text)

    recipe_name = name_match.group(1).strip() if name_match else ""
    calories = float(calories_match.group(1)) if calories_match else 0.0
    proteins = float(protein_match.group(1)) if protein_match else 0.0

    def extract_float(s: str) -> float:
        # only match numbers with at least one digit
        m = re.search(float_pattern, s)
        if not m:
            return 0.0
        try:
            return float(m.group(1))
        except ValueError:
            # in case the text is weird, fail soft
            return 0.0

    raw_lines = text.splitlines()

    ingredients = []
    in_ing = False
    last_ing_idx = None  # track index of last ingredient line

    for idx, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line:
            continue

        if line.lower().startswith("ingredients"):
            in_ing = True
            continue

        if in_ing and line.startswith("- "):
            last_ing_idx = idx  # remember where the last ingredient was seen

            body = line[2:]
            parts = [p.strip() for p in body.split(",")]
            if len(parts) < 4:
                continue
            name = parts[0]

            weight = extract_float(parts[1])
            total_cal = extract_float(parts[2])
            protein = extract_float(parts[3])

            ingredients.append(
                {
                    "ingredient": name,
                    "weight_g_or_ml": weight,
                    # you can recompute calories_per_unit later if needed
                    "total_calories": total_cal,
                    "protein": protein,
                }
            )

    # --- Detect extra trailing junk after last ingredient line ---
    has_extra_trailing_text = False
    extra_trailing_text = ""

    if last_ing_idx is not None and last_ing_idx + 1 < len(raw_lines):
        trailing_lines = [l for l in raw_lines[last_ing_idx + 1:] if l.strip()]
        if trailing_lines:
            has_extra_trailing_text = True
            extra_trailing_text = "\n".join(trailing_lines).strip()

    return {
        "recipe_name": recipe_name,
        "calories": calories,
        "proteins": proteins,
        "ingredients": ingredients,
        # structure cleanliness info:
        "has_extra_trailing_text": has_extra_trailing_text,
        "extra_trailing_text": extra_trailing_text,
    }


if __name__ == "__main__":
    data_path = Path("./../eval_outputs/rag_finetuned_1b.json")
    with data_path.open("r", encoding="utf-8") as f:
        qa_data = json.load(f)

    results = []
    for ex in qa_data:
        expected_raw = ex["answer"]
        model_raw = ex["model_answer"]

        expected_recipe = parse_recipe_text(expected_raw)
        model_recipe = parse_recipe_text(model_raw)

        results.append(
            {
                "query": ex["question"],
                # keep raw text so nothing is “lost”
                "expected_raw": expected_raw,
                "model_raw": model_raw,
                # parsed / structured views
                "expected_output": expected_recipe,
                "model_output": model_recipe,
            }
        )

    output_path = Path("./../eval_outputs/rag_finetuned_1b_json.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(results)} entries to {output_path}")