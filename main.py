from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ingestor import fetch_and_store_repo
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "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")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_6cMTJd_LsS7y1tjUmKhudxrPGnfcQSFPVozHnm4W7K5uq41XnFe7Gbkyt3x5gdyQiiZjzT")

model = SentenceTransformer('all-MiniLM-L6-v2')
index = None

class RepoRequest(BaseModel):
    repo_url: str

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Codebase QA is running!"}

@app.post("/load-repo")
def load_repo(req: RepoRequest):
    global index
    repo_name = req.repo_url.rstrip("/").split("/")[-2] + "-" + req.repo_url.rstrip("/").split("/")[-1]
    repo_name = repo_name.lower()[:45]
    index = fetch_and_store_repo(req.repo_url, GITHUB_TOKEN, PINECONE_API_KEY, repo_name)
    return {"message": "Repo loaded!"}

@app.post("/ask")
def ask(req: QuestionRequest):
    if index is None:
        return {"error": "No repo loaded yet!"}

    question_embedding = model.encode(req.question).tolist()

    results = index.query(
        vector=question_embedding,
        top_k=3,
        include_metadata=True
    )

    chunks = [r.metadata["text"] for r in results.matches]
    files = [r.metadata["file"] for r in results.matches]

    context = ""
    for chunk, file in zip(chunks, files):
        context += f"\n--- {file} ---\n{chunk}\n"

    prompt = f"""You are a code assistant. Answer the question using ONLY the code provided below.
Also mention which file the answer comes from.

Code context:
{context}

Question: {req.question}
"""

    client = Groq(api_key=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
)

    def stream():
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        yield f"\n||SOURCES||{json.dumps(list(set(files)))}"

    return StreamingResponse(stream(), media_type="text/plain")

@app.get("/architecture")
def get_architecture():
    if index is None:
        return {"error": "No repo loaded yet!"}

    results = index.query(
        vector=model.encode("import from module").tolist(),
        top_k=50,
        include_metadata=True
    )

    file_imports = {}
    for r in results.matches:
        file = r.metadata["file"]
        text = r.metadata["text"]
        if file not in file_imports:
            file_imports[file] = set()
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("from ") and " import " in line:
                module = line.split("from ")[1].split(" import ")[0].strip()
                file_imports[file].add(module)

    files = [f for f in file_imports.keys() if not f.startswith("tests/") and not f.startswith("examples/") and not f.startswith("docs/")]
    nodes = [{"id": f, "name": f.split("/")[-1]} for f in files]

    links = []
    for file, imports in file_imports.items():
        if file not in files:
            continue
        for imp in imports:
            for other_file in files:
                if file != other_file:
                    short_module = other_file.split("/")[-1].replace(".py", "")
                    if imp == short_module or imp.endswith("." + short_module) or short_module in imp:
                        links.append({"source": file, "target": other_file})

    links = list({(l["source"], l["target"]): l for l in links}.values())[:150]

    return {"nodes": nodes, "links": links}

@app.get("/interview")
def get_interview():
    if index is None:
        return {"error": "No repo loaded yet!"}

    question_embedding = model.encode("main functionality core logic important functions").tolist()
    results = index.query(
        vector=question_embedding,
        top_k=5,
        include_metadata=True
    )

    chunks = [r.metadata["text"] for r in results.matches]
    files = [r.metadata["file"] for r in results.matches]

    context = ""
    for chunk, file in zip(chunks, files):
        context += f"\n--- {file} ---\n{chunk}\n"

    prompt = f"""You are a technical interviewer. Based on the following code, generate 5 interview questions with answers.

Format your response as JSON like this:
[
  {{"question": "...", "answer": "...", "difficulty": "Easy/Medium/Hard"}},
  ...
]

Return ONLY the JSON array, nothing else.

Code:
{context}
"""

    client = Groq(api_key=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
)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    import re
    text = response.choices[0].message.content
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        questions = json.loads(match.group())
    else:
        questions = []

    return {"questions": questions}