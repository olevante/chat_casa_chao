import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


# =========================
# CONFIGURAÇÕES
# =========================

PASTA_PDFS = "arquivos"
ARQUIVO_FAISS = "rag/base_vetorial"


# =========================
# LISTA DE URLS
# =========================

lista_urls = [
    "https://www.instagram.com/acasachao/",
    "https://casachao.keepo.bio/",
    "https://www.booking.com/hotel/br/casa-chao.pt-br.html",
    "https://maps.app.goo.gl/YtxYhdsLe4mcJfpR7",
    "https://www.tripadvisor.ie/Hotel_Review-g3844154-d27140114-Reviews-Casa_Chao-Ponta_do_Corumbau_State_of_Bahia.html",
    "https://www.diggy.menu/casachao"
]


# =========================
# CARREGAR DOCUMENTOS
# =========================

def carregar_docs():

    docs = []

    # ---------- PDFs ----------
    for arquivo in os.listdir(PASTA_PDFS):
        if arquivo.endswith(".pdf"):
            caminho = os.path.join(PASTA_PDFS, arquivo)
            loader = PyPDFLoader(caminho)
            docs.extend(loader.load())

    # ---------- URLs ----------
    loader_web = WebBaseLoader(lista_urls)
    docs.extend(loader_web.load())

    return docs


# =========================
# CRIAR BASE VETORIAL
# =========================

def criar_base():

    print("Carregando documentos...")

    docs = carregar_docs()

    print("Dividindo em chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    print("Criando embeddings...")

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("Salvando base vetorial...")

    vectorstore.save_local(ARQUIVO_FAISS)

    print("✅ Base criada com sucesso!")


if __name__ == "__main__":
    criar_base()
