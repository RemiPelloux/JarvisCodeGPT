from typing import List, Tuple
from io import BytesIO
import zipfile
import os

from utils.docstore.document import Document
from utils.text_splitter import RecursiveCharacterTextSplitter
from utils.vectorstores import FAISS, VectorStore
from utils.vectorstores.faiss import FAISS
from utils.parse_utils import parse_docx, parse_pdf, parse_txt, parse_vtt




@st.experimental_memo()
def parse_docx(file: BytesIO) -> str:
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


@st.experimental_memo()
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)

        output.append(text)

    return output


@st.experimental_memo()
def parse_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


@st.experimental_memo()
def parse_vtt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove WebVTT header
    text = re.sub(r"^WEBVTT\s*\n", "", text, flags=re.IGNORECASE)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


@st.experimental_memo()
def parse_zipfile(file: BytesIO) -> List[Tuple[str, str]]:
    import zipfile
    import os

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
