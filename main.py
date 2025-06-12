# main.py
from llama_cpp import Llama
import streamlit as st

@st.cache_resource
def load_llama_model():
    return Llama(
        model_path="./models/llama/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx=512,
        n_threads=4,
        verbose=False
    )

llm = load_llama_model()

st.title("🦙 Mistral Chatbot (Offline)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask anything...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    prompt = f"[INST] {user_input} [/INST]"
    output = llm(prompt, max_tokens=200)
    response = output["choices"][0]["text"].strip()

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
