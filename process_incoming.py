import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import sys

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


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1:8b",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    data = r.json()

    if "response" not in data:
        print("LLM Error:", data)
        return ""

    return data["response"]


df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a question: ")
question_embedding = create_embedding([incoming_query])[0]


# similarity
similarities = cosine_similarity(
    np.vstack(df['embedding']),
    [question_embedding]
).flatten()

top_results = 5
top_indices = similarities.argsort()[::-1][:top_results]

new_df = df.iloc[top_indices]


prompt = f'''I am Doing a Project. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
# with open("prompt.txt", "w", encoding="utf-8") as f:
#     f.write(prompt)


response = inference(prompt)

print("\nFinal Answer:\n")
print(response)


with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)
