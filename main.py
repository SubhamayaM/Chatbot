# main.py
from llama_cpp import Llama
import streamlit as st
import os

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# -------------------------
# Load LLaMA model
# -------------------------
@st.cache_resource
def load_llama_model():
    return Llama(
        model_path="./models/llama/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx=512,
        n_threads=4,
        verbose=False
    )

llm = load_llama_model()

# -------------------------
# Load OCR model for documents
# -------------------------
@st.cache_resource
def load_ocr_model():
    return ocr_predictor(pretrained=True)

def extract_text_from_document(file_path):
    model = load_ocr_model()
    doc = DocumentFile.from_pdf(file_path) if file_path.endswith(".pdf") else DocumentFile.from_images(file_path)
    result = model(doc)
    return result.render()  # Returns HTML/text content

# -------------------------
# Streamlit UI
# -------------------------
st.title("🦙 Mistral Chatbot")

# Track chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# File Upload Section
# -------------------------
uploaded_file = st.file_uploader("📄 Upload a scanned document (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"✅ Uploaded: `{uploaded_file.name}`")
    st.session_state["uploaded_doc_path"] = file_path

    # Extract content from uploaded document
    st.markdown("### 🔍 Extracting content from document...")
    extracted_html = extract_text_from_document(file_path)

    st.markdown("### 📑 Extracted Document Content:")
    st.markdown(extracted_html, unsafe_allow_html=True)
    st.session_state["extracted_content"] = extracted_html

# -------------------------
# Chat Input Section
# -------------------------
user_input = st.chat_input("Ask anything...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Use document content if available
    if "extracted_content" in st.session_state:
        doc_text = st.session_state["extracted_content"]
        prompt = f"[INST] Document:\n{doc_text}\n\nQuestion: {user_input} [/INST]"
    else:
        prompt = f"[INST] {user_input} [/INST]"

    output = llm(prompt, max_tokens=200)
    response = output["choices"][0]["text"].strip()

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
