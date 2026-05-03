# Book Assistant

A RAG (Retrieval-Augmented Generation) chatbot that lets you converse with any PDF book. Built as an AI Engineer portfolio project.

## Architecture

```
PDF
 │
 ▼
ingest.py  ── PyMuPDF extracts text per page
           ── RecursiveCharacterTextSplitter (500 tok / 50 tok overlap)
           ── all-MiniLM-L6-v2 embeddings (local, sentence-transformers)
           ── ChromaDB persisted to ./chroma_db
                         │
                         ▼
app.py  ──── Streamlit chat UI
           ── User question → retriever.py fetches top-4 chunks
           ── chain.py builds LangChain retrieval chain
           ── Gemini 1.5 Flash generates answer from context
           ── Source page numbers displayed under every answer
```

**Key design choices:**
- Embeddings run fully locally — no OpenAI key or cost.
- Gemini 1.5 Flash is used on the free tier for generation.
- ChromaDB is persisted on disk; re-running `ingest.py` resets the store.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Google API key
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=<your key>
# Get a free key at https://aistudio.google.com/app/apikey

# 3. Ingest your PDF (run once)
python src/ingest.py data/mybook.pdf

# 4. Launch the app
streamlit run src/app.py
```

> Run all commands from the `book-assistant/` root directory.

## Project Structure

```
book-assistant/
├── data/            # Drop your PDF here (gitignored)
├── src/
│   ├── ingest.py    # PDF → chunks → embeddings → ChromaDB
│   ├── retriever.py # Loads ChromaDB, returns LangChain retriever
│   ├── chain.py     # RAG chain: retriever + Gemini prompt
│   └── app.py       # Streamlit UI
├── chroma_db/       # Auto-created on first ingest (gitignored)
├── .env.example
├── requirements.txt
└── README.md
```
