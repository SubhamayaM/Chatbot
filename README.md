# 🤖🦙 Chatbot using miniLLAMA
This project is a private, local AI assistant powered by MiniLLaMA (Mistral) using llama-cpp-python. It intelligently processes scanned documents (PDFs, images), extracts table-like structured text, retrieves the most relevant information, and can answer multiple questions or generate summaries using RAG (Retrieval-Augmented Generation).

Features:

💬 Offline Chatbot	Powered by MiniLLaMA via llama-cpp-python, no internet needed.

📄 PDF/Image Upload	Upload scanned documents, bills, reports, etc.

📊 Table-Aware OCR	Extracts structured table content using doctr (TableNet-style).

🔍 RAG	Uses SentenceTransformer + FAISS to find the most relevant document chunks.

📝 Text Summarizer	Type summarize to get a short summary of the entire document.

🔢 Multi-Question Support	Ask more than one question at a time — the bot will number answers.

🌐 Streamlit UI	Simple, web-based interface to upload, chat, and interact.

🔒 100% Offline	Secure, fast, and private — ideal for sensitive data.
