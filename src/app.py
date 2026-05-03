import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from src.chain import build_chain

st.set_page_config(page_title="Book Assistant", page_icon="📚", layout="centered")
st.title("📚 Book Assistant")
st.caption("Ask anything about your book — answers grounded in the text.")

if "chain" not in st.session_state:
    with st.spinner("Loading model and vector store…"):
        st.session_state.chain = build_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about the book…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = st.session_state.chain.invoke({"input": question})

        answer = result["answer"]
        source_docs = result["context"]

        pages = sorted({doc.metadata.get("page", "?") for doc in source_docs})
        page_str = ", ".join(str(p) for p in pages)

        st.markdown(answer)
        st.caption(f"Sources — page(s): {page_str}")

    full_response = f"{answer}\n\n*Sources — page(s): {page_str}*"
    st.session_state.messages.append({"role": "assistant", "content": full_response})
