import streamlit as st
from components.sidebar import sidebar
from openai.error import OpenAIError
from utils import (
    embed_docs,
    get_answer,
    get_sources,
    parse_docx,
    parse_pdf,
    parse_txt,
    parse_vtt,
    search_docs,
    text_to_docs,
    wrap_text_in_html,
)


def clear_submit():
    st.session_state["submit"] = False


st.set_page_config(page_title="InsideGPT", page_icon="📖", layout="wide")
st.header("InsideGPT 📖")

if "lang" not in st.session_state:
    st.session_state["lang"] = "fr"

sidebar()

uploaded_file = st.file_uploader(
    "Téléchargez un fichier pdf, docx, vtt, txt",
    type=["pdf", "docx", "txt", "vtt"],
    help="Vous pouvez télécharger un fichier pdf,docx ,vtt ou txt.",
    on_change=clear_submit,
)

index = None
doc = None
if uploaded_file is not None:
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
    try:
        with st.spinner("Indexation du document... ⏳"):
            index = embed_docs(text)
        st.session_state["api_key_configured"] = True
    except OpenAIError as e:
        st.error(e._message)

query = st.text_area("Poser votre question ..", on_change=clear_submit)
with st.expander("Options Avancées"):
    show_all_chunks = st.checkbox("Montrer tous les morceaux de réponse")
    show_full_doc = st.checkbox("Afficher le contenu analysé du document")


if show_full_doc and doc:
    with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_text_in_html(doc)}</p>", unsafe_allow_html=True)

button = st.button("Envoi")
if button or st.session_state.get("submit"):
    if not st.session_state.get("api_key_configured"):
        st.error("Configurez votre clé OpenAI API !")
    elif not index:
        st.error("Veuillez télécharger un document !")
    elif not query:
        st.error("Veuillez poser une question ?")
    else:
        st.session_state["submit"] = True
        # Output Columns
        answer_col, sources_col = st.columns(2)
        sources = search_docs(index, query)

        try:
            answer = get_answer(sources, query)
            if not show_all_chunks:
                # Get the sources for the answer
                sources = get_sources(answer, sources)

            with answer_col:
                st.markdown(print(st.session_state["lang"]))
                st.markdown("#### Réponse")
                st.markdown(answer["output_text"].split("SOURCES: ")[0])

            with sources_col:
                st.markdown("#### Sources")
                for source in sources:
                    st.markdown(source.page_content)
                    st.markdown(source.metadata["source"])
                    st.markdown("---")

        except OpenAIError as e:
            st.error(e._message)
