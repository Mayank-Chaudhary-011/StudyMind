# StudyMind AI — Chat with your PDFs

> A full Retrieval-Augmented Generation (RAG) pipeline. Upload your PDFs, ask questions in plain English, and auto-generate quizzes — grounded in your own material.

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://studymind--ai.streamlit.app/)
[![Landing Page](https://img.shields.io/badge/Landing%20Page-Vercel-000000?style=for-the-badge&logo=vercel)](https://landing-page-study-mind-ai.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge)](https://langchain.com)

---

## What is StudyMind AI?

StudyMind AI is an open-source RAG pipeline built for students and learners. Instead of relying on what an LLM memorised during training, it retrieves relevant chunks directly from your own PDFs — so every answer is grounded in your actual material, never hallucinated.

**Key numbers:**
- 384-dimensional embeddings (all-MiniLM-L6-v2)
- 512 tokens per chunk, 64-token overlap
- Top-5 chunks retrieved per query
- Unlimited PDFs supported

---

## Features

| Feature | Description |
|---|---|
| **Chat with PDFs** | Upload any PDF and ask questions in plain English |
| **Auto Quiz Generator** | Generate scored MCQs from your material instantly |
| **Multi-PDF Knowledge Base** | Upload multiple PDFs — all embedded into one ChromaDB store |
| **Source Citations** | Every answer shows exactly which chunks were used |
| **Local Embeddings** | sentence-transformers runs locally — zero API cost for embeddings |
| **Persistent Vector Store** | ChromaDB persists to disk — upload once, query forever |

---

## How It Works

```
PDF Files
    ↓
loader.py       → PyPDFLoader loads PDFs as LangChain Document objects
    ↓
chunker.py      → RecursiveCharacterTextSplitter (512 tokens, 64 overlap)
    ↓
embedder.py     → sentence-transformers (all-MiniLM-L6-v2, 384-dim, local)
    ↓
vector_store.py → ChromaDB persists vectors to disk, deduplicates on every run
    ↓
retriever.py    → Cosine similarity search → Top-5 most relevant chunks
    ↓
generator.py    → Groq LLaMA 3 generates grounded answer from retrieved context
```

Every step is a **separate Python file** — modular, readable, easy to extend.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| RAG Framework | LangChain |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (persistent) |
| LLM | Groq API (LLaMA 3) |
| PDF Loading | PyPDFLoader |
| UI | Streamlit |
| Deployment | Streamlit Cloud |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Mayank-Chaudhary-011/StudyMind.git
cd StudyMind
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key
Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
Get a free key at [console.groq.com](https://console.groq.com)

### 4. Run the app
```bash
streamlit run main.py
```

---

## Project Structure

```
StudyMind/
├── main.py              # Streamlit UI entry point
├── loader.py            # PDF loading with PyPDFLoader
├── chunker.py           # RecursiveCharacterTextSplitter
├── embedder.py          # Local sentence-transformers embeddings
├── vector_store.py      # ChromaDB persistent vector store
├── retriever.py         # Cosine similarity retrieval (top-5)
├── generator.py         # Groq LLM answer generation
├── requirements.txt
└── .streamlit/
    └── secrets.toml     # API keys (not committed)
```

---

## Roadmap

- [ ] Conversation memory (multi-turn chat)
- [ ] Reranking with Cohere
- [ ] Web URL ingestion
- [ ] Agentic RAG with LangGraph
- [ ] Study analytics dashboard
- [ ] Fully local mode (Ollama)

---

## Built By

**Mayank Chaudhary** — AI Engineer  
[GitHub](https://github.com/Mayank-Chaudhary-011) · [LinkedIn](https://linkedin.com/in/mayank-chaudhary-977828232) · chaudharymayank996@gmail.com
