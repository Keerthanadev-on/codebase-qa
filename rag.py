from groq import Groq
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def answer_question(question: str, index, groq_api_key: str):
    # Step 1 - Embed the question
    question_embedding = model.encode(question).tolist()

    # Step 2 - Search Pinecone
    results = index.query(
        vector=question_embedding,
        top_k=3,
        include_metadata=True
    )

    # Step 3 - Build context
    chunks = [r.metadata["text"] for r in results.matches]
    files = [r.metadata["file"] for r in results.matches]

    context = ""
    for chunk, file in zip(chunks, files):
        context += f"\n--- {file} ---\n{chunk}\n"

    # Step 4 - Ask Groq
    client = Groq(api_key=groq_api_key)

    prompt = f"""You are a code assistant. Answer the question using ONLY the code provided below.
Also mention which file the answer comes from.

Code context:
{context}

Question: {question}
"""

    def stream():
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return {
        "stream": stream,
        "sources": list(set(files))
    }