from typing import List
from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
from utils.openai_embed import OpenAIEmbeddings
import streamlit as st


@st.experimental_memo()
def embed_docs(text: str | List[str]) -> VectorStore:
    """Embeds a list of documents and returns a FAISS index"""

    docs = text_to_docs(text)

    if not st.session_state.get("OPENAI_API_KEY"):
        raise AuthenticationError(
            "Authentication error. Please enter a valid OpenAI API key."
        )

    # Embed the chunks
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.get("OPENAI_API_KEY"))
    index = FAISS.from_documents(docs, embeddings)

    return index


@st.cache(allow_output_mutation=True)
def text_to_docs(text: str | List[str]) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""

    if isinstance(text, str):
        # Take a single string as one page
        text = [text]

    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        chunk_size = 800
        separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        chunk_overlap = 0

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, separators=separators, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={"page": doc.metadata["page"], "chunk": i},
            )

            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)

    return doc_chunks
