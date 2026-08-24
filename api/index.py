import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum

app = FastAPI()

# Allow CORS for local and web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get current script path to find embeddings.json and jsons/ relative to it
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "embeddings.json")
JSONS_DIR = os.path.join(BASE_DIR, "jsons")

# Load embeddings database in memory
embeddings_db = []
if os.path.exists(DB_PATH):
    with open(DB_PATH, "r", encoding="utf-8") as f:
        embeddings_db = json.load(f)

# Pure Python Cosine Similarity (no numpy/scikit-learn required for Vercel)
def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v):
    return sum(x * x for x in v) ** 0.5

def calculate_similarity(v1, v2):
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product(v1, v2) / (mag1 * mag2)

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
def run_query(req: QueryRequest):
    if not embeddings_db:
        raise HTTPException(status_code=500, detail="Embeddings database 'embeddings.json' not found or empty.")

    # Get Ollama endpoint from environment variables (for Vercel deployment)
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    # 1. Embed query
    try:
        r = requests.post(f"{ollama_host}/api/embed", json={
            "model": "mxbai-embed-large",
            "input": [req.query]
        }, headers={"bypass-tunnel-reminder": "true"}, timeout=10)
        r.raise_for_status()
        question_embedding = r.json()["embeddings"][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding via Ollama: {e}")

    # 2. Similarity search
    scored_chunks = []
    for chunk in embeddings_db:
        if chunk.get("embedding"):
            sim = calculate_similarity(chunk["embedding"], question_embedding)
            # Remove embedding vector from output to save bandwidth
            output_chunk = {k: v for k, v in chunk.items() if k != "embedding"}
            output_chunk["similarity"] = sim
            scored_chunks.append(output_chunk)

    # Sort and take top 5
    scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    top_chunks = scored_chunks[:5]

    # 3. LLM Inference
    prompt = f'''I am Doing a Project. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{json.dumps(top_chunks, ensure_ascii=False)}
---------------------------------
"{req.query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
    try:
        r = requests.post(f"{ollama_host}/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }, headers={"bypass-tunnel-reminder": "true"}, timeout=30)
        r.raise_for_status()
        response_text = r.json()["response"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query LLM via Ollama: {e}")

    return {
        "response": response_text,
        "chunks": top_chunks
    }

@app.get("/api/transcripts")
def get_transcripts():
    if not os.path.exists(JSONS_DIR):
        return []
    
    # List and sort all json files numerically
    def get_prefix_num(filename):
        try:
            return int(filename.split(' ')[0])
        except ValueError:
            return 999
            
    files = sorted([f for f in os.listdir(JSONS_DIR) if f.endswith(".json")], key=get_prefix_num)
    return files

@app.get("/api/transcript/{filename}")
def get_transcript(filename: str):
    file_path = os.path.join(JSONS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Transcript file not found.")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Handler for AWS Lambda / Vercel Serverless
handler = Mangum(app)
