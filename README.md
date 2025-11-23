# Health Coach 🧠🏋️‍♀️

LLM-powered personal health, exercise, and nutrition assistant.

## Features
- Personalized exercise suggestions based on goals.
- Meal ideas and calorie information.
- Fine-tuned Gemma-3 model with LoRA adapters.

## Project Structure
- `backend/` – API server (e.g., FastAPI).
- `frontend/` – Web/mobile UI.
- `models/` – Training & inference scripts (LoRA fine-tuning, loading).
- `data/` – Raw and processed datasets (see `data/README.md`).
- `notebooks/` – Exploration and data generation.
- `outputs/` – Local model artifacts (ignored by git).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate    # on macOS/Linux
pip install -r requirements.txt
```
## Training LoRA

```python models/Lora_FineTune.py```
