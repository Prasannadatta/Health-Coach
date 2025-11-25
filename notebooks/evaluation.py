from datasets import load_dataset
import json
import math
import re
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
import pandas as pd


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Convert value to float, with a safe default."""
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def ingredient_name_set(recipe: Dict[str, Any]) -> Set[str]:
    """Return a lower-cased set of ingredient names."""
    names: Set[str] = set()
    for ing in recipe.get("ingredients", []):
        name = str(ing.get("ingredient", "")).strip().lower()
        if name:
            names.add(name)
    return names


def compute_totals_from_ingredients(recipe: Dict[str, Any]) -> Dict[str, float]:
    """Compute total calories and protein by summing ingredients."""
    total_cal = 0.0
    total_protein = 0.0
    for ing in recipe.get("ingredients", []):
        total_cal += _safe_float(ing.get("total_calories", 0.0))
        total_protein += _safe_float(ing.get("protein", 0.0))
    return {
        "calories_from_ingredients": total_cal,
        "proteins_from_ingredients": total_protein,
    }


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


# ---------- Query constraint parsing ----------

CAL_CONSTRAINT_RE = re.compile(
    r"\b(?P<kind>under|below|less than|at most|no more than|"
    r"over|at least|more than|min(?:imum)?|above|"
    r"around|near|about|between)\s+"
    r"(?P<n1>\d+(\.\d+)?)"
    r"(?:\s*(?:and|-|to)\s*(?P<n2>\d+(\.\d+)?))?"
    r"\s*(?:calories|calorie|kcal)?",
    re.IGNORECASE,
)

PROTEIN_CONSTRAINT_RE = re.compile(
    r"\b(?P<kind>at least|over|more than|min(?:imum)?|"
    r"under|below|less than|at most|no more than|above)\s+"
    r"(?P<n1>\d+(\.\d+)?)\s*(?:g|grams)?\s*(?:protein)?",
    re.IGNORECASE,
)


def parse_calorie_constraint(question: str) -> Optional[Dict[str, Any]]:
    """
    Extract a calorie constraint from the question, if any.

    Returns dict like:
      {"type": "max", "value": 600}
      {"type": "min", "value": 400}
      {"type": "around", "value": 500}
      {"type": "between", "low": 400, "high": 600}
    or None.
    """
    m = CAL_CONSTRAINT_RE.search(question)
    if not m:
        return None

    kind = m.group("kind").lower()
    n1 = float(m.group("n1"))
    n2 = m.group("n2")
    if n2 is not None:
        n2 = float(n2)

    if kind in ["under", "below", "less than", "at most", "no more than"]:
        return {"type": "max", "value": n1}
    if kind in ["over", "at least", "more than", "min", "minimum", "above"]:
        return {"type": "min", "value": n1}
    if kind in ["around", "near", "about"]:
        return {"type": "around", "value": n1}
    if kind == "between" and n2 is not None:
        low, high = sorted([n1, n2])
        return {"type": "between", "low": low, "high": high}

    # Fallback: treat as "around"
    return {"type": "around", "value": n1}


def parse_protein_constraint(question: str) -> Optional[Dict[str, Any]]:
    """
    Extract a protein constraint from the question, if any.

    Returns dict like:
      {"type": "min", "value": 25}
      {"type": "max", "value": 30}
    or None.
    """
    m = PROTEIN_CONSTRAINT_RE.search(question)
    if not m:
        return None

    kind = m.group("kind").lower()
    n1 = float(m.group("n1"))

    if kind in ["under", "below", "less than", "at most", "no more than"]:
        return {"type": "max", "value": n1}
    if kind in ["over", "at least", "more than", "min", "minimum", "above"]:
        return {"type": "min", "value": n1}

    return None


def check_calorie_constraint(pred_cal: float, constraint: Dict[str, Any],
                             rel_margin: float = 0.15) -> bool:
    """Return True if predicted calories satisfy the query constraint (with margin)."""
    if constraint is None:
        return True

    t = constraint["type"]
    if t == "max":
        return pred_cal <= constraint["value"] * (1 + rel_margin)
    if t == "min":
        return pred_cal >= constraint["value"] * (1 - rel_margin)
    if t == "around":
        v = constraint["value"]
        low = v * (1 - rel_margin)
        high = v * (1 + rel_margin)
        return low <= pred_cal <= high
    if t == "between":
        low = constraint["low"] * (1 - rel_margin)
        high = constraint["high"] * (1 + rel_margin)
        return low <= pred_cal <= high
    return True


def check_protein_constraint(pred_prot: float, constraint: Dict[str, Any],
                             rel_margin: float = 0.20, abs_margin: float = 5.0) -> bool:
    """Return True if predicted protein satisfies the query constraint (with margin)."""
    if constraint is None:
        return True

    t = constraint["type"]
    v = constraint["value"]
    eff_margin = max(abs_margin, v * rel_margin)

    if t == "max":
        return pred_prot <= v + eff_margin
    if t == "min":
        return pred_prot >= v - eff_margin
    return True


# ---------- Dietary constraints (vegan, vegetarian, etc.) ----------

MEAT_FISH = {
    "chicken", "beef", "pork", "lamb", "mutton", "fish", "salmon", "tuna",
    "shrimp", "prawn", "bacon", "ham", "sausage", "turkey"
}

DAIRY = {
    "milk", "cheese", "butter", "yogurt", "cream", "paneer", "ghee", "ice cream"
}

EGG = {"egg", "eggs", "egg white", "egg whites", "egg yolk", "egg yolks"}

PEANUT = {"peanut", "peanuts", "groundnut", "groundnuts"}

TREE_NUTS = {
    "almond", "almonds", "walnut", "walnuts", "cashew", "cashews", "pistachio",
    "pistachios", "pecan", "pecans", "hazelnut", "hazelnuts", "macadamia",
    "nut", "nuts"
}

# NEW: gluten sources
GLUTEN_GRAINS = {
    "wheat", "barley", "rye", "farro", "spelt",
    "bulgur", "couscous", "semolina", "seitan",
    "pasta", "bread", "tortilla", "flour"
}


def detect_dietary_constraints(question: str) -> Dict[str, bool]:
    q = question.lower()
    return {
        "vegan_required": "vegan" in q,
        "vegetarian_required": "vegetarian" in q and "vegan" not in q,
        "egg_free_required": "egg-free" in q or "egg free" in q,
        "dairy_free_required": "dairy-free" in q or "dairy free" in q,
        "peanut_free_required": "peanut-free" in q or "peanut free" in q,
        "tree_nut_free_required": "tree-nut-free" in q or "tree nut free" in q,
        "gluten_free_required": "gluten-free" in q or "gluten free" in q,
    }


def contains_any_keyword(names: Set[str], keywords: Set[str]) -> bool:
    for name in names:
        for kw in keywords:
            if kw in name:
                return True
    return False


def check_dietary_satisfaction(names_pred: Set[str],
                               flags: Dict[str, bool]) -> Dict[str, bool]:
    vegan_ok = True
    if flags["vegan_required"]:
        if contains_any_keyword(names_pred, MEAT_FISH | DAIRY | EGG | PEANUT | TREE_NUTS):
            vegan_ok = False

    vegetarian_ok = True
    if flags["vegetarian_required"]:
        if contains_any_keyword(names_pred, MEAT_FISH):
            vegetarian_ok = False

    egg_free_ok = True
    if flags["egg_free_required"]:
        if contains_any_keyword(names_pred, EGG):
            egg_free_ok = False

    dairy_free_ok = True
    if flags["dairy_free_required"]:
        if contains_any_keyword(names_pred, DAIRY):
            dairy_free_ok = False

    peanut_free_ok = True
    if flags["peanut_free_required"]:
        if contains_any_keyword(names_pred, PEANUT):
            peanut_free_ok = False

    tree_nut_free_ok = True
    if flags["tree_nut_free_required"]:
        if contains_any_keyword(names_pred, TREE_NUTS):
            tree_nut_free_ok = False

    gluten_free_ok = True
    if flags.get("gluten_free_required", False):
        if contains_any_keyword(names_pred, GLUTEN_GRAINS):
            gluten_free_ok = False

    all_required = [
        (flags["vegan_required"], vegan_ok),
        (flags["vegetarian_required"], vegetarian_ok),
        (flags["egg_free_required"], egg_free_ok),
        (flags["dairy_free_required"], dairy_free_ok),
        (flags["peanut_free_required"], peanut_free_ok),
        (flags["tree_nut_free_required"], tree_nut_free_ok),
        (flags.get("gluten_free_required", False), gluten_free_ok),
    ]
    any_required = any(req for req, _ in all_required)
    all_satisfied = all(ok for req, ok in all_required if req)

    return {
        "vegan_satisfied_pred": vegan_ok,
        "vegetarian_satisfied_pred": vegetarian_ok,
        "egg_free_satisfied_pred": egg_free_ok,
        "dairy_free_satisfied_pred": dairy_free_ok,
        "peanut_free_satisfied_pred": peanut_free_ok,
        "tree_nut_free_satisfied_pred": tree_nut_free_ok,
        "gluten_free_satisfied_pred": gluten_free_ok,
        "any_dietary_constraint_required": any_required,
        "all_dietary_constraints_satisfied_pred": (all_satisfied if any_required else True),
    }


# ---------- Main evaluation ----------

def evaluate_recipe_pair(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    question: str = "",
    calories_tol_pct: float = 0.10,
    protein_tol_abs: float = 5.0,
    sum_cal_tol_pct: float = 0.10,
    sum_cal_tol_abs: float = 50.0,
    sum_prot_tol_abs: float = 5.0,
) -> Dict[str, Any]:
    """
    Compare expected vs actual recipe outputs and return metrics.

    expected / actual can either be:
      - the recipe dict itself, or
      - a dict with key "answer" containing the recipe dict.

    Now also:
      - adds top-level vs ingredient-sum consistency flags
      - checks calorie/protein constraints from the question
      - checks simple dietary constraints (vegan, vegetarian, etc.)
      - adds structure metrics (format_ok, extra text, etc.)
    """
    gt = expected.get("answer", expected)
    pred = actual.get("answer", actual)

    # Top-level values
    c_gt = _safe_float(gt.get("calories", 0.0))
    p_gt = _safe_float(gt.get("proteins", 0.0))
    c_pred = _safe_float(pred.get("calories", 0.0))
    p_pred = _safe_float(pred.get("proteins", 0.0))

    # Ingredient-derived totals
    gt_totals = compute_totals_from_ingredients(gt)
    pred_totals = compute_totals_from_ingredients(pred)

    # Errors vs expected (top-level)
    calories_abs_error = abs(c_pred - c_gt)
    protein_abs_error = abs(p_pred - p_gt)

    calories_rel_error = (
        calories_abs_error / c_gt if c_gt > 0 else math.nan
    )
    protein_rel_error = (
        protein_abs_error / p_gt if p_gt > 0 else math.nan
    )

    # Check if within tolerance (compared to expected)
    within_calories_tol = (
        calories_rel_error <= calories_tol_pct
        if not math.isnan(calories_rel_error)
        else False
    )
    within_protein_tol = protein_abs_error <= protein_tol_abs

    # Ingredient-name overlap
    names_gt = ingredient_name_set(gt)
    names_pred = ingredient_name_set(pred)
    ingredient_jaccard = jaccard(names_gt, names_pred)

    # Consistency of each recipe's own totals vs ingredients (sum consistency)
    gt_cal_consistency_err = abs(gt_totals["calories_from_ingredients"] - c_gt)
    gt_protein_consistency_err = abs(gt_totals["proteins_from_ingredients"] - p_gt)
    pred_cal_consistency_err = abs(pred_totals["calories_from_ingredients"] - c_pred)
    pred_protein_consistency_err = abs(pred_totals["proteins_from_ingredients"] - p_pred)

    gt_calories_sum_consistent = gt_cal_consistency_err <= max(
        sum_cal_tol_abs, c_gt * sum_cal_tol_pct
    ) if c_gt > 0 else False

    pred_calories_sum_consistent = pred_cal_consistency_err <= max(
        sum_cal_tol_abs, c_pred * sum_cal_tol_pct
    ) if c_pred > 0 else False

    gt_protein_sum_consistent = gt_protein_consistency_err <= sum_prot_tol_abs
    pred_protein_sum_consistent = pred_protein_consistency_err <= sum_prot_tol_abs

    # ---- Structure / format metrics ----
    num_ingredients_gt = len(gt.get("ingredients", []))
    num_ingredients_pred = len(pred.get("ingredients", []))

    format_ok_pred = (
        bool(pred.get("recipe_name")) and
        c_pred > 0 and
        p_pred > 0 and
        num_ingredients_pred > 0
    )

    has_extra_text_pred = bool(pred.get("has_extra_text", False))

    # ---- Query-based constraints ----
    cal_constraint = parse_calorie_constraint(question) if question else None
    prot_constraint = parse_protein_constraint(question) if question else None
    dietary_flags = detect_dietary_constraints(question) if question else {
        "vegan_required": False,
        "vegetarian_required": False,
        "egg_free_required": False,
        "dairy_free_required": False,
        "peanut_free_required": False,
        "tree_nut_free_required": False,
        "gluten_free_required": False,
    }
    dietary_satisfaction = check_dietary_satisfaction(names_pred, dietary_flags)

    calorie_constraint_satisfied_pred = (
        check_calorie_constraint(c_pred, cal_constraint)
        if cal_constraint is not None else True
    )
    protein_constraint_satisfied_pred = (
        check_protein_constraint(p_pred, prot_constraint)
        if prot_constraint is not None else True
    )

    # Overall "all query constraints" flag for predicted recipe
    any_numeric_constraint = (cal_constraint is not None or prot_constraint is not None)
    all_numeric_constraints_ok = (
        calorie_constraint_satisfied_pred and protein_constraint_satisfied_pred
    )
    any_dietary_required = dietary_satisfaction["any_dietary_constraint_required"]
    all_dietary_ok = dietary_satisfaction["all_dietary_constraints_satisfied_pred"]

    if any_numeric_constraint or any_dietary_required:
        all_query_constraints_satisfied_pred = all_numeric_constraints_ok and all_dietary_ok
    else:
        # no constraints → trivially satisfied
        all_query_constraints_satisfied_pred = True

    return {
        # Question for context
        "question": question,

        # Names
        "recipe_name_gt": gt.get("recipe_name"),
        "recipe_name_pred": pred.get("recipe_name"),

        # Top-level calories / protein
        "calories_gt": c_gt,
        "calories_pred": c_pred,
        "proteins_gt": p_gt,
        "proteins_pred": p_pred,

        # Errors between expected and actual
        "calories_abs_error": calories_abs_error,
        "calories_rel_error": calories_rel_error,
        "protein_abs_error": protein_abs_error,
        "protein_rel_error": protein_rel_error,

        # Tolerance checks vs expected
        "within_calories_tolerance": within_calories_tol,
        "within_protein_tolerance": within_protein_tol,

        # Ingredient-derived totals
        "calories_from_ingredients_gt": gt_totals["calories_from_ingredients"],
        "proteins_from_ingredients_gt": gt_totals["proteins_from_ingredients"],
        "calories_from_ingredients_pred": pred_totals["calories_from_ingredients"],
        "proteins_from_ingredients_pred": pred_totals["proteins_from_ingredients"],

        # Consistency (how well each recipe's totals match its own ingredients)
        "gt_calories_consistency_error": gt_cal_consistency_err,
        "gt_protein_consistency_error": gt_protein_consistency_err,
        "pred_calories_consistency_error": pred_cal_consistency_err,
        "pred_protein_consistency_error": pred_protein_consistency_err,

        # Sum-consistency booleans
        "gt_calories_sum_consistent": gt_calories_sum_consistent,
        "pred_calories_sum_consistent": pred_calories_sum_consistent,
        "gt_protein_sum_consistent": gt_protein_sum_consistent,
        "pred_protein_sum_consistent": pred_protein_sum_consistent,

        # Ingredient overlap
        "ingredient_jaccard": ingredient_jaccard,
        "ingredients_gt": sorted(names_gt),
        "ingredients_pred": sorted(names_pred),

        # Structure metrics
        "num_ingredients_gt": num_ingredients_gt,
        "num_ingredients_pred": num_ingredients_pred,
        "format_ok_pred": format_ok_pred,
        "has_extra_text_pred": has_extra_text_pred,

        # Query constraints (numeric)
        "calorie_constraint": cal_constraint,
        "protein_constraint": prot_constraint,
        "calorie_constraint_satisfied_pred": calorie_constraint_satisfied_pred,
        "protein_constraint_satisfied_pred": protein_constraint_satisfied_pred,

        # Query constraints (dietary)
        **dietary_flags,
        **dietary_satisfaction,

        # Overall flag
        "all_query_constraints_satisfied_pred": all_query_constraints_satisfied_pred,
    }


def load_qa_pairs(path: str):
    """Load qa_pairs.json and return the full list."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_answers(path: str):
    """
    Load qa_pairs.json and return a list of only the `answer` dicts.
    Each element looks like:
      {
        "recipe_name": ...,
        "calories": ...,
        "proteins": ...,
        "ingredients": [...]
      }
    """
    qa_pairs = load_qa_pairs(path)
    answers = [entry["answer"] for entry in qa_pairs]
    return answers


if __name__ == "__main__":

    dataset_path = Path("./../eval_outputs/rag_finetuned_1b_json.json")
    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    stats_list = []

    for ex in dataset:
        expected = ex["expected_output"]
        actual = ex["model_output"]
        question = ex.get("question", "")

        stats = evaluate_recipe_pair(expected, actual, question=question)
        stats_list.append(stats)

    df = pd.DataFrame(stats_list)

    out_path = Path("./../eval_outputs/rag_finetuned_1b.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved metrics for {len(df)} examples to {out_path}")
    # print(df.head())