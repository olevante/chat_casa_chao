from rag.busca import buscar_contexto
import streamlit as st
from dotenv import load_dotenv
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# =========================
# CONFIG INICIAL
# =========================

load_dotenv()

st.set_page_config(
    page_title="Chat Casa Chão",
    page_icon="assets/user_3.png",
    layout="wide"
)

# =========================
# CACHE DE PERSONA
# =========================

@st.cache_data
def carregar_persona():
    try:
        with open("persona.txt", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "Você é um assistente virtual da Casa Chão."


# =========================
# MEMÓRIA POR SESSÃO
# =========================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "historico" not in st.session_state:
    st.session_state.historico = ChatMessageHistory()


def get_session_history(session_id):
    return st.session_state.historico


# =========================
# CACHE DO LLM
# =========================

@st.cache_resource
def carregar_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        streaming=True
    )


llm = carregar_llm()

# =========================
# PROMPT
# =========================

persona = carregar_persona()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", persona + """

Você possui acesso a um banco de conhecimento abaixo:

{contexto}

Use PRIORITARIAMENTE essas informações.

Se o contexto não tiver a resposta:
responda normalmente usando seu conhecimento geral.
"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)

# =========================
# CHAIN
# =========================

chain = prompt | llm

chain_com_memoria = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# =========================
# HEADER
# =========================

col1, col2 = st.columns([1, 6])

with col1:
    st.image("assets/logocasachao.png", width=80)

with col2:
    st.header("Bem-vindo à Casa Chão", divider=True)
    st.caption("Seu guia virtual para Corumbau")

# =========================
# HISTÓRICO
# =========================

for msg in st.session_state.historico.messages:
    if msg.type == "human":
        with st.chat_message("user", avatar="assets/user_1.png"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant", avatar="assets/user_3.png"):
            st.markdown(msg.content)

# =========================
# INPUT
# =========================

pergunta = st.chat_input("Pergunte sobre a Casa Chão...")

if pergunta:

    with st.chat_message("user", avatar="assets/user_1.png"):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar="assets/user_3.png"):

        resposta_container = st.empty()
        resposta_final = ""

        try:
            contexto_rag = buscar_contexto(pergunta)

            for chunk in chain_com_memoria.stream(
                {
                    "input": pergunta,
                    "contexto": contexto_rag
                },
                config={"configurable": {"session_id": st.session_state.session_id}}
            ):
                resposta_final += chunk.content
                resposta_container.markdown(resposta_final)

            # SALVAR HISTÓRICO APENAS UMA VEZ
            st.session_state.historico.add_user_message(pergunta)
            st.session_state.historico.add_ai_message(resposta_final)


        except Exception:
            st.error("⚠️ Erro ao gerar resposta. Tente novamente.")

