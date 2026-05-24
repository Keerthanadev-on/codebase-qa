from github import Github, Auth
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import time

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_pinecone_index(pinecone_api_key: str, index_name: str):
    pc = Pinecone(api_key=pinecone_api_key)
    
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)
    
    return pc.Index(index_name)

def fetch_and_store_repo(repo_url: str, github_token: str, pinecone_api_key: str, index_name: str):
    index = get_pinecone_index(pinecone_api_key, index_name)
    
    # Check if already indexed
    stats = index.describe_index_stats()
    if stats.total_vector_count > 0:
        print("Loaded from cache! ✅")
        return index

    g = Github(auth=Auth.Token(github_token))
    parts = repo_url.rstrip("/").split("/")
    full_repo_name = f"{parts[-2]}/{parts[-1]}"
    repo = g.get_repo(full_repo_name)
    print(f"Connected to repo: {repo.full_name}")

    code_extensions = [".py", ".js", ".java", ".ts", ".cpp", ".c", ".go"]
    contents = repo.get_contents("")
    code_files = []

    while contents:
        file = contents.pop(0)
        if file.type == "dir":
            contents.extend(repo.get_contents(file.path))
        elif any(file.path.endswith(ext) for ext in code_extensions):
            code_files.append(file)

    print(f"Found {len(code_files)} code files")

    vectors = []
    for file in code_files:
        try:
            content = file.decoded_content.decode("utf-8")
            lines = content.split("\n")
            chunks = ["\n".join(lines[i:i+50]) for i in range(0, len(lines), 50)]

            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    embedding = model.encode(chunk).tolist()
                    vectors.append({
                        "id": f"{file.path}_chunk{i}",
                        "values": embedding,
                        "metadata": {"file": file.path, "text": chunk[:1000]}
                    })

            if len(vectors) >= 100:
                index.upsert(vectors=vectors)
                vectors = []

            print(f"Stored: {file.path}")
        except Exception as e:
            print(f"Skipped {file.path}: {e}")

    if vectors:
        index.upsert(vectors=vectors)

    print("\nDone! ✅")
    return index