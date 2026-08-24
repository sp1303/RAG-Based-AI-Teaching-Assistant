import os
import sys
import subprocess
import json
import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import whisper

# Force UTF-8 encoding for standard output on Windows
sys.stdout.reconfigure(encoding='utf-8')


# -----------------------------
# SETUP DIRECTORIES
# -----------------------------
os.makedirs("audios", exist_ok=True)
os.makedirs("jsons", exist_ok=True)

# Find the first video (Lecture 1)
video_folder = "SnapTube Video"
if not os.path.exists(video_folder):
    print(f"Error: {video_folder} directory does not exist.")
    exit(1)

video_files = os.listdir(video_folder)
test_video = None
for f in video_files:
    if f.startswith("Lec - 1_") and f.endswith(".mp4"):
        test_video = f
        break

if not test_video:
    # fallback to any mp4
    for f in video_files:
        if f.endswith(".mp4"):
            test_video = f
            break

if not test_video:
    print("Error: No mp4 video files found to test.")
    exit(1)

print(f"Found test video file: {test_video}")

# -----------------------------
# STEP 1: VIDEO TO MP3
# -----------------------------
# Extract tutorial number and name similar to Video_to_Mp3.py
tutorial_number = test_video.split("_")[0].replace("Lec - ", "").replace("Lec -", "").strip()
file_name = test_video.split("_", 1)[1]
name_without_ext = os.path.splitext(file_name)[0].strip()

mp3_filename = f"{tutorial_number}_{name_without_ext}.mp3"
mp3_path = os.path.join("audios", mp3_filename)

if not os.path.exists(mp3_path):
    print(f"Converting video to MP3: {mp3_path}...")
    subprocess.run([
        "ffmpeg",
        "-i", os.path.join(video_folder, test_video),
        mp3_path
    ], check=True)
    print("MP3 conversion complete.")
else:
    print(f"MP3 file already exists at {mp3_path}")

# -----------------------------
# STEP 2: MP3 TO JSON (Whisper)
# -----------------------------
clean_name = f"{tutorial_number} {name_without_ext}".replace("(720P_HD)", "").strip()
json_path = os.path.join("jsons", f"{clean_name}.json")

if not os.path.exists(json_path):
    print("Loading Whisper 'tiny' model (using tiny for speed on CPU)...")
    # Using tiny instead of large-v2 for fast CPU testing
    model = whisper.load_model("tiny")
    print("Transcribing audio...")
    result = model.transcribe(
        audio=mp3_path,
        language="hi",
        task="translate",
        word_timestamps=False
    )
    
    # create chunks
    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "number": tutorial_number,
            "title": name_without_ext.replace("(720P_HD)", "").strip(),
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })
        
    print(f"Saving transcribed chunks to {json_path}...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "chunks": chunks,
            "text": result["text"]
        }, f, ensure_ascii=False, indent=2)
else:
    print(f"JSON transcript already exists at {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]

# -----------------------------
# STEP 3: READ CHUNKS & EMBED
# -----------------------------
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "mxbai-embed-large",
        "input": text_list
    })
    r.raise_for_status()
    data = r.json()
    return data["embeddings"]

print("Generating embeddings for chunks...")
texts = [c['text'] for c in chunks]
embeddings = create_embedding(texts)

my_dict = []
for i, chunk in enumerate(chunks):
    chunk['chunk_id'] = i
    chunk['embedding'] = embeddings[i]
    my_dict.append(chunk)

df = pd.DataFrame.from_records(my_dict)
joblib.dump(df, "embeddings.joblib")
print("Saved embeddings to embeddings.joblib")

# -----------------------------
# STEP 4: INFERENCE (RAG Query)
# -----------------------------
incoming_query = "What is a data warehouse and what are some examples?"
print(f"\nRunning test query: '{incoming_query}'")

question_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(
    np.vstack(df['embedding']),
    [question_embedding]
).flatten()

top_results = min(5, len(df))
top_indices = similarities.argsort()[::-1][:top_results]
new_df = df.iloc[top_indices]

prompt = f'''I am Doing a Project. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''

print("Sending query to Ollama (llama3.2)...")
r = requests.post("http://localhost:11434/api/generate", json={
    "model": "llama3.2",
    "prompt": prompt,
    "stream": False
})
r.raise_for_status()
response_data = r.json()
response = response_data["response"]

print("\n=== FINAL RAG RESPONSE ===")
print(response)
print("==========================")
