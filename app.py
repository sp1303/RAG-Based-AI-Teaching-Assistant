import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

# Configure Streamlit page
st.set_page_config(
    page_title="RAG AI Teaching Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design, Micro-Animations, and Typography
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Animation for chat messages */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stChatMessage {
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        border-radius: 12px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    
    /* Expander Hover Animations */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        border-radius: 10px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        margin-bottom: 8px !important;
    }
    
    div[data-testid="stExpander"]:hover {
        border-color: #8A2BE2 !important;
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(138, 43, 226, 0.1);
    }
    
    /* Styled Title & Header elements */
    .app-title {
        background: linear-gradient(135deg, #8A2BE2, #4A90E2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .app-subtitle {
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glow effect on inputs */
    textarea {
        transition: border-color 0.3s, box-shadow 0.3s !important;
    }
    textarea:focus {
        border-color: #8A2BE2 !important;
        box-shadow: 0 0 10px rgba(138, 43, 226, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Render Styled Header
st.markdown('<div class="app-title">🎓 RAG AI Teaching Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Ask questions about video lectures and trace exact timestamps instantly.</div>', unsafe_allow_html=True)

# Cache database loading
@st.cache_resource
def load_db():
    return joblib.load('embeddings.joblib')

try:
    df = load_db()
except FileNotFoundError:
    st.error("Error: 'embeddings.joblib' not found. Please run the transcription pipeline first.")
    st.stop()

def create_embedding(text_list):
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "mxbai-embed-large",
            "input": text_list
        })
        return r.json().get("embeddings", [])
    except Exception as e:
        st.error(f"Ollama connection error: {e}")
        return []

def inference(prompt):
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        })
        return r.json().get("response", "")
    except Exception as e:
        st.error(f"Ollama connection error: {e}")
        return ""

# Initialize session state keys
if "messages" not in st.session_state:
    st.session_state.messages = []
if "matched_chunks" not in st.session_state:
    st.session_state.matched_chunks = None

# Build a set of matched (video number, start time) to highlight them in the browser
matched_set = set()
if st.session_state.matched_chunks is not None:
    for idx, row in st.session_state.matched_chunks.iterrows():
        matched_set.add((str(row['number']).strip(), round(float(row['start']), 2)))

# Sidebar layout
st.sidebar.title("📚 Course Content & Sources")
sidebar_mode = st.sidebar.radio("Choose Mode:", ["🔍 Search Results Context", "📖 Browse All Transcripts"])

if sidebar_mode == "🔍 Search Results Context":
    st.sidebar.subheader("Retrieved Source Chunks")
    if st.session_state.matched_chunks is not None:
        for idx, row in st.session_state.matched_chunks.iterrows():
            start_min = int(row['start'] // 60)
            start_sec = int(row['start'] % 60)
            end_min = int(row['end'] // 60)
            end_sec = int(row['end'] % 60)
            
            with st.sidebar.expander(
                f"🎯 MATCH | Video {row['number']}: {row['title']} ({start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d})"
            ):
                st.info(row['text'])
    else:
        st.sidebar.info("Ask a question to view the retrieved video chunks here.")

else:
    st.sidebar.subheader("Browse Video Transcripts")
    if os.path.exists("jsons"):
        # Sort files numerically by prefix number
        def get_prefix_num(filename):
            try:
                return int(filename.split(' ')[0])
            except ValueError:
                return 999
        
        json_files = sorted([f for f in os.listdir("jsons") if f.endswith(".json")], key=get_prefix_num)
        
        if json_files:
            selected_file = st.sidebar.selectbox("Select a video:", json_files)
            if selected_file:
                with open(os.path.join("jsons", selected_file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Show full text
                st.sidebar.markdown("**Full Translated Text:**")
                st.sidebar.text_area("Transcript text", data.get("text", ""), height=200)
                
                # Expand to show timestamped chunks
                st.sidebar.markdown("**Individual Timestamped Chunks:**")
                for chunk in data.get("chunks", []):
                    start_min = int(chunk['start'] // 60)
                    start_sec = int(chunk['start'] % 60)
                    end_min = int(chunk['end'] // 60)
                    end_sec = int(chunk['end'] % 60)
                    
                    # Check if this chunk is part of the active search results context
                    is_match = (str(chunk['number']).strip(), round(float(chunk['start']), 2)) in matched_set
                    
                    if is_match:
                        with st.sidebar.expander(
                            f"🎯 MATCH: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}", expanded=True
                        ):
                            st.success(chunk['text'])
                    else:
                        with st.sidebar.expander(f"⏳ {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"):
                            st.write(chunk['text'])
        else:
            st.sidebar.info("No JSON transcripts found in 'jsons' folder.")
    else:
        st.sidebar.info("No 'jsons' directory found.")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if incoming_query := st.chat_input("Ask a question about the course..."):
    # Display user message
    with st.chat_message("user"):
        st.write(incoming_query)
    st.session_state.messages.append({"role": "user", "content": incoming_query})

    # RAG Retrieval
    with st.spinner("Searching course videos..."):
        emb = create_embedding([incoming_query])
        if not emb:
            st.stop()
        
        question_embedding = emb[0]
        similarities = cosine_similarity(
            np.vstack(df['embedding']),
            [question_embedding]
        ).flatten()
        
        top_indices = similarities.argsort()[::-1][:5]
        new_df = df.iloc[top_indices]
        
        # Save matched chunks to session state for the sidebar
        st.session_state.matched_chunks = new_df
        
        prompt = f'''I am Doing a Project. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
        response = inference(prompt)

    # Display assistant response and rerun to update sidebar
    if response:
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
