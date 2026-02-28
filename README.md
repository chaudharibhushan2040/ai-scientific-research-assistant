# 🧠 AI Scientific Research Assistant

A production-level Retrieval-Augmented Generation (RAG) system built using:

- LLaMA 3 (Groq)
- LangChain
- FAISS
- HuggingFace Embeddings
- Streamlit

---

## 🚀 Features

- Multi-PDF Upload
- Semantic Search using FAISS
- Scientific Question Answering
- Executive Document Summary
- Source References
- Download Answers as PDF
- Token Usage Tracking
- Dark Scientific UI

---

## 🏗 Architecture

PDF → Text Chunking → Embeddings → FAISS → Retrieval → LLaMA → Response

---

## 🛠 Installation (Local)

```bash
git clone https://github.com/chaudharibhushan2040/ai-scientific-research-assistant/edit/main/README.md
cd ai-scientific-research-assistant
pip install -r requirements.txt
streamlit run app.py

---

## 🔐 Environment Variable

Create a `.env` file in the root directory:

GROQ_API_KEY=your_api_key_here

---

## 👨‍💻 Author

Bhushan Chaudhari
AI & GenAI Enthusiast

