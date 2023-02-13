import streamlit as st
from langchain.prompts import PromptTemplate


def set_language(lang):
    st.session_state["lang"] = lang


def set_template(lang):
    template_fr = """Créez une réponse finale en francais aux questions données en utilisant les extraits de documents fournis (dans n'importe quel ordre) comme références. Incluez toujours une section "SOURCES" dans votre réponse ne comprenant que le minimum de sources nécessaires pour répondre à la question. Si vous ne pouvez pas répondre à la question, indiquez simplement que vous ne savez pas. N'essayez pas de fabriquer une réponse et laissez la section SOURCES vide.

    ---------

    QUESTION: What  is the purpose of ARPA-H?
    =========
    Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
    Source: 1-32
    Content: While we’re at it, let’s make sure every American can get the health care they need. \n\nWe’ve already made historic investments in health care. \n\nWe’ve made it easier for Americans to get the care they need, when they need it. \n\nWe’ve made it easier for Americans to get the treatments they need, when they need them. \n\nWe’ve made it easier for Americans to get the medications they need, when they need them.
    Source: 1-33
    Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat’s why I’m calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
    Source: 1-30
    =========
    FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
    SOURCES: 1-32

    ---------

    QUESTION: {question}
    =========
    {summaries}
    =========
    REPONSE FINALE:
    """
    template_en = """Create a final answer to the given questions using the provided document excerpts(in no particular order) as references. ALWAYS include a "SOURCES" section in your answer including only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not know. Do not attempt to fabricate an answer and leave the SOURCES section empty.
    ---------
    QUESTION: What  is the purpose of ARPA-H?
    =========
    Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
    Source: 1-32
    Content: While we’re at it, let’s make sure every American can get the health care they need. \n\nWe’ve already made historic investments in health care. \n\nWe’ve made it easier for Americans to get the care they need, when they need it. \n\nWe’ve made it easier for Americans to get the treatments they need, when they need them. \n\nWe’ve made it easier for Americans to get the medications they need, when they need them.
    Source: 1-33
    Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat’s why I’m calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
    Source: 1-30
    =========
    FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
    SOURCES: 1-32
    ---------
    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER:"""
    template_it = """ Creare una risposta finale alle domande date utilizzando come riferimenti gli estratti dei documenti forniti (in ordine sparso). Includete SEMPRE una sezione "FONTI" nella vostra risposta, includendo solo il minimo di fonti necessarie per rispondere alla domanda. Se non siete in grado di rispondere alla domanda, dichiarate semplicemente che non lo sapete. Non cercate di inventare una risposta lasciando vuota la sezione FONTI.

    ---------
    QUESTION: What  is the purpose of ARPA-H?
    =========
    Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
    Source: 1-32
    Content: While we’re at it, let’s make sure every American can get the health care they need. \n\nWe’ve already made historic investments in health care. \n\nWe’ve made it easier for Americans to get the care they need, when they need it. \n\nWe’ve made it easier for Americans to get the treatments they need, when they need them. \n\nWe’ve made it easier for Americans to get the medications they need, when they need them.
    Source: 1-33
    Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat’s why I’m calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
    Source: 1-30
    =========
    FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
    SOURCES: 1-32
    ---------
    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER:"""

    if lang == "francais":
        st.session_state["template"] = template_fr
        st.session_state["template_rdy"] = PromptTemplate(
            template=st.session_state["template"], input_variables=["summaries", "question"]
        )
    elif lang == "english":
        st.session_state["template"] = template_en
        st.session_state["template_rdy"] = PromptTemplate(
            template=st.session_state["template"], input_variables=["summaries", "question"]
        )
    elif lang == "italiano":
        st.session_state["template"] = template_it
        st.session_state["template_rdy"] = PromptTemplate(
            template=st.session_state["template"], input_variables=["summaries", "question"]
        )
    else:
        st.session_state["template"] = template_en
        st.session_state["template_rdy"] = PromptTemplate(
            template=st.session_state["template"], input_variables=["summaries", "question"]
        )


def set_openai_api_key(api_key: str):
    st.session_state["OPENAI_API_KEY"] = api_key


def sidebar():
    with st.sidebar:
        st.markdown("# A Propos")
        st.markdown(
            "📖InsideGPT permet de poser des questions sur vos "
            "documents et d'obtenir des réponses précises avec des citations instantanées. "
            ""
        )
        st.markdown(
            "Cet outil est en cours de développement. "
            "Mod par [RemiPelloux](https://github.com/RemiPelloux)"
        )
        st.markdown("---")
        st.markdown(
            "## Comment utiliser\n"
            "1. Entrez votre [OpenAI API key](https://platform.openai.com/account/api-keys) ci-dessous 🔑\n"
            "2. Téléchargez un fichier pdf, docx, txt, vtt, pptx 📄\n"
            "3. Posez une question sur le document upload💬\n"
        )
        api_key_input = st.text_input(
            "Cle OpenAI API",
            type="password",
            placeholder="Collez votre clé OpenAI API ici (sk-...)",
            help="Vous pouvez obtenir votre clé API à partir de https://platform.openai.com/account/api-keys.",
            value=st.session_state.get("OPENAI_API_KEY", ""),
        )

        if api_key_input:
            set_openai_api_key(api_key_input)

        st.markdown("### Langue")
        lang = st.selectbox("Choisir une langue", ["francais", "english", "italiano"])
        if lang != st.session_state["lang"]:
            set_language(lang)
            set_template(lang)
        st.markdown("---")

