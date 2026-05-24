# Codebase QA 🌸

Ask anything about any GitHub codebase in plain English.

**Live Demo:** [codebase-qa-app.netlify.app](https://codebase-qa-app.netlify.app)

---

## What it does

Paste a GitHub repo → ask questions → get answers based on the actual source code with file citations.

---

## Features

- AI chat with streaming responses
- File citations on every answer
- Architecture visualization
- Interview question generator
- Persistent caching (index once, load instantly)

---

## Tech Stack

Python · FastAPI · Groq (LLaMA 3.3) · Pinecone · Sentence Transformers · GitHub API

---

## Run locally

```bash
git clone https://github.com/Keerthanadev-on/codebase-qa.git
cd codebase-qa
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
python -m http.server 5500
```

Add your API keys in `main.py`:
- `GITHUB_TOKEN`
- `GROQ_API_KEY`
- `PINECONE_API_KEY`

---

Built by **G Keerthana** · [GitHub](https://github.com/Keerthanadev-on)
