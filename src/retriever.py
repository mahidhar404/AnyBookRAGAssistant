import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
EMBED_MODEL = "models/gemini-embedding-001"
TOP_K = 4


def get_retriever() -> VectorStoreRetriever:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": TOP_K})
