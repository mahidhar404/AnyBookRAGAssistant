import os
import sys

# Allow `from src.retriever import ...` when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.retriever import get_retriever

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that answers questions strictly based on "
    "the provided book excerpts. Cite relevant details when possible. "
    "If the context does not contain enough information, say so honestly.\n\n"
    "Context:\n{context}"
)


def build_chain() -> Runnable:
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(get_retriever(), qa_chain)
