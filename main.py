# main.py
from llama_cpp import Llama
import streamlit as st
import os
import faiss
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from sentence_transformers import SentenceTransformer

# -------------------------
# Load LLaMA model
# -------------------------
@st.cache_resource
def load_llama_model():
    return Llama(
        model_path="./models/llama/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )

llm = load_llama_model()

# -------------------------
# For Loading OCR model
# -------------------------
@st.cache_resource
def load_ocr_model():
    return ocr_predictor(pretrained=True)

def extract_text_from_document(file_path):
    model = load_ocr_model()
    doc = DocumentFile.from_pdf(file_path) if file_path.endswith(".pdf") else DocumentFile.from_images(file_path)
    result = model(doc)
    return result.render()

# -------------------------
# For Loading embedding model
# -------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# -------------------------
# Streamlit UI (User Interface)
# -------------------------
st.title("🦙🤖 Mistral Chatbot (Offline)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
    st.session_state.doc_embeddings = None
    st.session_state.index = None

# -------------------------
# Uploading Document Prompt
# -------------------------
uploaded_file = st.file_uploader("📄 Upload Intel Report (PDF / Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"✅ Uploaded: `{uploaded_file.name}`")

    extracted = extract_text_from_document(file_path)
    st.session_state["full_doc"] = extracted

    st.markdown("### 🧾 Extracted Intel:")
    st.markdown(extracted, unsafe_allow_html=True)

    chunk_size = 300
    chunks = [extracted[i:i + chunk_size] for i in range(0, len(extracted), chunk_size)]
    st.session_state.doc_chunks = chunks

# FAISS - Vector Database 
    
    embeddings = embedder.encode(chunks)
    st.session_state.doc_embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(st.session_state.doc_embeddings.shape[1])
    index.add(st.session_state.doc_embeddings)
    st.session_state.index = index

# -------------------------
# RAG context retrieval
# -------------------------
def retrieve_context(question, top_k=3):
    question_embedding = embedder.encode([question]).astype("float32")
    D, I = st.session_state.index.search(question_embedding, top_k)
    return "\n\n".join([st.session_state.doc_chunks[i] for i in I[0]])

# -------------------------
# Chat Interface
# -------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question or type 'summarize last 24 hours'...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if "summarize" in user_input.lower():
        if "24" in user_input.lower() or "last 24" in user_input.lower():
            context = "\n\n".join(st.session_state.doc_chunks)
            prompt = (
                "You are an intelligence analyst assistant. "
                "Generate a concise and structured summary of key events, movements, and messages "
                "reported in the last 24 hours from the following content:\n\n"
                f"{context}\n\n"
                "Format the summary with timestamps or bullet points where possible."
            )
        else:
            context = "\n\n".join(st.session_state.doc_chunks)
            prompt = f"Summarize the following document:\n\n{context}\n\nSummary:"
    else:
        if st.session_state.get("index"):
            context = retrieve_context(user_input)
            prompt = (
                "You are a helpful assistant. Please answer ALL questions asked. "
                "If there are multiple questions, number your answers clearly.\n\n"
                f"Context:\n{context}\n\nQuestions:\n{user_input}\n\nAnswer:"
            )
        else:
            prompt = user_input

    output = llm(prompt[:6000], max_tokens=300)
    response = output["choices"][0]["text"].strip()

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
