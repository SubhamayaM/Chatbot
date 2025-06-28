# 🤖🦙 Mistral Chatbot (Offline)
An advanced, fully offline AI assistant built for secure document analysis and question answering, powered by Mistral. Ideal for defense, intelligence, and mission-critical workflows, it can read scanned documents, extract structured data, summarize field reports, and answer complex multi-part questions — all with RAG (Retrieval-Augmented Generation) and OCR-based table extraction.

Features:

🧠 Mistral LLM (Offline)	Runs locally via llama-cpp-python using .gguf model.

📄 Document OCR (doctr)	Extracts text and tables from scanned PDFs/images.

🧾 Table-Aware Summarization	Understands layout and formats structured summaries.

🔍 RAG (Retrieval-Augmented Generation)	Retrieves relevant text chunks using FAISS + SentenceTransformer.

📝 Smart Summarization	Type summarize or summarize last 24 hours for SITREP.

🔢 Multi-Question Handling	Responds to multiple queries in one prompt with numbered answers.

💬 Streamlit UI	Secure, interactive chat-style interface.

🔒 Fully Offline & Private	No cloud, no data leaks — air-gapped capable.
