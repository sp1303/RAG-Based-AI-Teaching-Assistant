import requests
import json
import os
import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib   

# Force UTF-8 encoding for standard output on Windows
sys.stdout.reconfigure(encoding='utf-8')


def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "mxbai-embed-large",
        "input": text_list
    })

    data = r.json()

    if "embeddings" not in data:
        print("Error response:", data)
        return []

    return data["embeddings"]

jsons = os.listdir("jsons")

my_dict = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}", "r", encoding="utf-8") as f:
        content = json.load(f)
    print(f"Creating embeddings for {json_file}") 

    texts = [c['text'] for c in content.get('chunks', [])]
    embeddings = create_embedding(texts)

    for i, chunk in enumerate(content.get("chunks", [])):
        chunk['chunk_id'] = chunk_id

        if i < len(embeddings):
            chunk['embedding'] = embeddings[i]
        else:
            chunk['embedding'] = None  # safety fallback

        chunk_id += 1
        my_dict.append(chunk)
        
    

   

df = pd.DataFrame.from_records(my_dict)
joblib.dump(df, "embeddings.joblib")
print("Successfully generated and saved embeddings.joblib!")
