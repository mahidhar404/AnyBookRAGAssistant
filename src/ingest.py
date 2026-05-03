"""
Run once to chunk, embed, and store a PDF into ChromaDB.
Usage: python src/ingest.py data/mybook.pdf
"""

import os
import sys
import shutil
import time

from dotenv import load_dotenv
load_dotenv()

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
EMBED_MODEL = "models/gemini-embedding-001"
CHUNK_SIZE = 500      # tokens
CHUNK_OVERLAP = 50    # tokens
EMBED_BATCH_SIZE = 80   # free tier: 100 embed requests/min; stay safely under
EMBED_RATE_SLEEP = 65   # seconds to wait between batches


def load_pdf(path: str) -> list[Document]:
    doc = fitz.open(path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append(Document(
                page_content=text,
                metadata={"page": page_num, "source": os.path.basename(path)},
            ))
    doc.close()
    return pages


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt2",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def embed_and_store(chunks: list[Document]) -> None:
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Cleared existing ChromaDB.")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    db = None
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        if db is None:
            db = Chroma.from_documents(batch, embeddings, persist_directory=CHROMA_PATH)
        else:
            db.add_documents(batch)
        done = min(i + EMBED_BATCH_SIZE, len(chunks))
        print(f"  Embedded {done}/{len(chunks)} chunks")
        if done < len(chunks):
            print(f"  Rate-limit pause ({EMBED_RATE_SLEEP}s)…")
            time.sleep(EMBED_RATE_SLEEP)

    print(f"Stored {len(chunks)} chunks → {CHROMA_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/ingest.py data/mybook.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"Loading: {pdf_path}")
    docs = load_pdf(pdf_path)
    print(f"  {len(docs)} pages extracted")

    chunks = chunk_documents(docs)
    print(f"  {len(chunks)} chunks created")

    embed_and_store(chunks)
    print("Ingestion complete.")
