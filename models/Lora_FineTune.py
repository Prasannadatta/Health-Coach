# models/Lora_FineTune.py

import os
import json
import random
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ========= CONFIG =========
MODEL_ID = "google/gemma-3-270m"


# 👉 EDIT THESE paths to match your data files
DATA_FILES = [
    "data/exercise_1.jsonl",   # Q/A in JSONL
    "data/Nutrition_1.json",  # Q/A in JSON array or dict
]

VAL_FRACTION = 0.1  # 10% for validation


# ========= DATA LOADING (no pyarrow, fully manual) =========
def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data can be a list of {question, answer} or a dict with a list inside
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # try to find the first list value that looks like examples
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    raise ValueError(f"Don't know how to interpret JSON structure in {path}")


def load_all_examples():
    all_rows = []
    for path in DATA_FILES:
        if not os.path.exists(path):
            print(f"⚠️  Warning: {path} does not exist, skipping.")
            continue

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".jsonl":
                rows = read_jsonl(path)
            elif ext == ".json":
                rows = read_json(path)
            else:
                print(f"⚠️  Warning: unsupported extension {ext} for {path}, skipping.")
                continue

            for r in rows:
                # normalize to {question, answer}
                if "question" in r and "answer" in r:
                    all_rows.append(
                        {"question": r["question"], "answer": r["answer"]}
                    )
                elif "input" in r and "output" in r:
                    all_rows.append(
                        {"question": r["input"], "answer": r["output"]}
                    )
                else:
                    # skip weird rows
                    continue
        except Exception as e:
            print(f"⚠️  Warning: failed to read {path}: {e}")

    if not all_rows:
        raise RuntimeError("No examples loaded. Check DATA_FILES paths & format.")
    return all_rows


def make_dataset():
    rows = load_all_examples()
    random.shuffle(rows)

    n_total = len(rows)
    n_val = max(1, int(VAL_FRACTION * n_total))

    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    print(f"Train examples: {len(train_rows)}")
    print(f"Val examples  : {len(val_rows)}")

    ds_train = Dataset.from_list(train_rows)
    ds_val = Dataset.from_list(val_rows)

    return DatasetDict({"train": ds_train, "val": ds_val})


# ========= TOKENIZER & MODEL =========
def make_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def make_model(device: str):
    # bfloat16 on mps, float32 on cpu
    dtype = torch.bfloat16 if device == "mps" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,  # modern transformers prefers `dtype` over `torch_dtype`
    )
    # Trainer will move model to device, but this is harmless:
    model.to(device)
    return model


# ========= PROMPT FORMATTING =========
def format_example(example):
    """
    Convert one row {question, answer} into a single training string.
    You can tweak the system style here.
    """
    user_q = example["question"]
    coach_a = example["answer"]

    text = (
        "You are a supportive, non-judgmental health and wellness coach. "
        "You give realistic, safe advice about nutrition, exercise, and daily habits.\n\n"
        f"User: {user_q}\n\n"
        f"Coach: {coach_a}"
    )
    return text


def formatting_func(example):
    # TRL calls this with ONE row: {"question": "...", "answer": "..."}
    return format_example(example)


# ========= MAIN =========
def main():
    # ----- Device -----
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Using device:", device)

    # ----- Data -----  
    dataset = make_dataset()

    # ----- Tokenizer & Model -----
    tokenizer = make_tokenizer()
    model = make_model(device)

    # ----- LoRA Config -----
    lora_config = LoraConfig(
        r=16,              # 8–32; higher = more capacity, more VRAM
        lora_alpha=32,     # usually 2 * r
        lora_dropout=0.05, # 0.0–0.1 typical
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # ----- Training Config (SFTConfig, not TrainingArguments) -----
    sft_config = SFTConfig(
        output_dir="lora-gemma270m-base",
        do_train=True,
        do_eval=True,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch = 8
        num_train_epochs=2.0,            # 1–3 usually enough for small data
        learning_rate=2e-4,              # 1e-4–3e-4 good range
        weight_decay=0.02,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        report_to=None,   # no wandb/tensorboard

        # SFT-specific
        max_length=512,   # truncate long Q/A at 512 tokens
        packing=False,

        # Device helpers
        use_mps_device=(device == "mps"),
        fp16=False,
        bf16=False,       # we already set dtype on model; leave these False
        remove_unused_columns=False,
    )

    # ----- SFT Trainer -----
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        peft_config=lora_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        formatting_func=formatting_func,
        processing_class=tokenizer,  # <- replaces old `tokenizer=` kwarg
    )

    # ----- Train -----
    trainer.train()

    # ----- Save Adapter -----
    save_dir = "lora-gemma270m-base-adapter"
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Saved LoRA adapter to {save_dir}")


if __name__ == "__main__":
    main()