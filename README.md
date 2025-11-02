# Health & Wellness Coach

LLM-powered wellness companion that generates daily meals, workouts, and supportive check-ins based on goals, diet, equipment, and mood. It combines rule-based health logic (BMR/TDEE, macros, exercise filters) with a conversational AI layer (LangChain/LangGraph-ready) so plans stay accurate **and** human.

---

## ✨ What it does

- Collect user profile (age, height, weight, goal, diet, equipment, injuries)
- Compute target calories/macros using standard formulas
- Select meals/exercises from public datasets
- Ask an LLM to turn that into an empathetic daily plan
- Let user say “I ate something else / I skipped / I’m tired”
- Adjust tone or plan based on mood + adherence

**Goal:** bridge the gap between number-only fitness apps and real coaching.

---

## 🏗 Repo structure

```text
health-wellness-coach/
  backend/    # API, rule engine, LLM calls, data access
  frontend/   # React web app (dashboard + chat)
  data/       # seed nutrition / exercise JSONs
  docs/       # prompts, API notes, diagrams
  README.md

Later:

  mobile/     # Expo / React Native app


⸻

🧠 Architecture idea
	1.	Rules first → BMR/TDEE, macros, equipment/injury filters
	2.	LLM second → format, explain, be supportive
	3.	Loop → user logs → LLM re-plans / encourages → store

This makes it easy to plug in:
	•	LangChain for tool-calling (getUser, getTodayPlan, logMeal)
	•	LangGraph for a small coaching state machine (low mood → softer plan)

⸻

🚀 Getting started

1. Clone

git clone https://github.com/<your-username>/health-wellness-coach.git
cd health-wellness-coach

2. Backend (Node)

cd backend
npm install
npm run dev   # or: node src/index.js

Create .env:

PORT=4000
OPENAI_API_KEY=your_key_here

3. Frontend (React)

cd ../frontend
npm install
npm run dev

Create frontend/.env:

VITE_API_URL=http://localhost:4000


⸻

🛠 Planned stack
	•	Backend: Node.js, Express, LangChain / LangGraph
	•	Frontend: React (Vite, MUI)
	•	DB: Postgres or MongoDB
	•	LLM: OpenAI-compatible endpoint

⸻

📂 data/

Put cleaned public datasets here:
	•	data/nutrition.json
	•	data/exercises.json

Backend can load these first before moving to a real DB.

⸻

📘 docs/

Keep:
	•	prompt templates (plan / adjust / coach)
	•	API design
	•	model notes

⸻

✅ .gitignore

node_modules/
.env
.env.*
dist/
build/
.DS_Store
.expo/


⸻

📄 License

MIT

