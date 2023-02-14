import streamlit as st
from typing import Any, Dict, List, Tuple
from io import BytesIO
import zipfile
import os

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, VectorStore
from langchain.vectorstores.faiss import FAISS
from openai import OpenAIError
from openai.error import AuthenticationError

from components.sidebar import sidebar
from utils import parse_docx, parse_pdf, parse_txt, parse_vtt
from utils import embed_docs, search_docs, get_answer, get_sources, wrap_text_in_html
from prompts import PromptTemplate


@st.cache(allow_output_mutation=True)
def parse_zipfile(file: BytesIO) -> List[Tuple[str, str]]:
	z = zipfile.ZipFile(file)
	file_list = []
	for name in z.namelist():
		if not os.path.isdir(name):
			with z.open(name) as f:
				if name.endswith(".pdf"):
					file_list.append((name, parse_pdf(f)))
				elif name.endswith(".docx"):
					file_list.append((name, parse_docx(f)))
				elif name.endswith(".txt"):
					file_list.append((name, parse_txt(f)))
	return file_list


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
		text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=800,
			separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
			chunk_overlap=0,
		)
		chunks = text_splitter.split_text(doc.page_content)
		for i, chunk in enumerate(chunks):
			doc = Document(
				page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
			)
			# Add sources a metadata
			doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
			doc_chunks.append(doc)
	return doc_chunks


@st.cache(allow_output_mutation=True)
def get_index(docs: List[Document]) -> VectorStore:
	"""Embeds a list of Documents and returns a FAISS index"""

	if not st.session_state.get("OPENAI_API_KEY"):
		raise AuthenticationError(
			"Erreur d'authentification. Veuillez entrer une clé API OpenAI valide."
		)
	else:
		# Embed the chunks
		embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.get("OPENAI_API_KEY"))  # type: ignore
		index = FAISS.from_documents(docs, embeddings)

		return index


if __name__ == "__main__":
	st.set_page_config(page_title="JarvisCodeGPT", page_icon="🚀", layout="wide")
	st.title("JarvisCodeGPT")

	sidebar()

	uploaded_file = st.file_uploader("Télécharger un fichier ou un fichier zip")

	if uploaded_file is not None:
		if uploaded_file.name.endswith(".zip"):
			file_list = parse_zipfile(uploaded_file)
			text = [x[1] for x in file_list]
			try:
				with st.spinner("Indexation du document... ⏳"):
					index = embed_docs(text)
				st.session_state["api_key_configured"] = True
			except OpenAIError as e:
				st.error(e._message)

			st.write(f"Le fichier zip contient les fichiers suivants: {', '.join([x[0] for x in file_list])}")
		else:
			if uploaded_file.name.endswith(".pdf"):
				doc = parse_pdf(uploaded_file)
			elif uploaded_file.name.endswith(".docx"):
				doc = parse_docx(uploaded_file)
			elif uploaded_file.name.endswith(".txt"):
				doc = parse_txt(uploaded_file)
			elif uploaded_file.name.endswith(".vtt"):
				doc = parse_vtt(uploaded_file)
			else:
				raise ValueError("Type de fichier non pris en charge")

			text = text_to_docs(doc)

			st.write(f"Titre du fichier: {uploaded_file.name}")

			try:
				with st.spinner("Indexation du document... ⏳"):
					index = embed_docs(text)
				st.session_state["api_key_configured"] = True
			except OpenAIError as e:
				st.error(e._message)

	query = st.text_input("Poser une question :")

	if st.button("Rechercher"):
		with st.spinner("Recherche de réponse... ⏳"):
			docs = search_docs(index, query)
			answer = get_answer(docs, query)

		st.write("Réponse:")
		st.write(wrap_text_in_html(answer["output_text"]))

		source_docs = get_sources(answer, docs)

		if source_docs:
			st.write("Sources:")
			for doc in source_docs:
				st.write(f"Page {doc.metadata['page']}, Chunk {doc.metadata['chunk']}:\n{doc.page_content}")
		else:
			st.write("Aucune source trouvée.")
