from typing import Any, Dict, List
from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
import streamlit as st
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from openai import OpenAI


def search_docs(index: VectorStore, query: str) -> List[Document]:
    """Searches a FAISS index for similar chunks to the query
    and returns a list of Documents."""

    # Search for similar chunks
    docs = index.similarity_search(query, k=5)
    return docs


def get_answer(docs: List[Document], query: str) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    # Get the answer
    chain = load_qa_with_sources_chain(
        OpenAI(
            temperature=0,
            openai_api_key=st.session_state.get("OPENAI_API_KEY"),
        ),
        chain_type="stuff",
        prompt=st.session_state["template_rdy"],
    )

    # Cohere doesn't work very well as of now.
    # chain = load_qa_with_sources_chain(Cohere(temperature=0), chain_type="stuff", prompt=STUFF_PROMPT)  # type: ignore
    answer = chain(
        {"input_documents": docs, "question": query},
        return_only_outputs=True,
    )
    return answer


def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""

    # Get sources for the answer
    source_keys = [
        s
        for s in answer["output_text"]
        .split("SOURCES: ")[-1]
        .split(", ")
    ]

    source_docs = []
    for doc in docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs


def wrap_text_in_html(text: str | List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])
