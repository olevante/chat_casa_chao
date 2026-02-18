import os
os.environ["USER_AGENT"] = "CasaChaoBot/1.0"

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

# =========================
# LISTA DE SITES
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
# FUNÇÕES
# =========================

def carrega_sites(lista_urls):
    documentos = []

    for url in lista_urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            documentos.extend(docs)
            print(f"✅ Carregado: {url}")
        except Exception as e:
            print(f"❌ Erro em {url}: {e}")

    texto = "\n\n".join([doc.page_content for doc in documentos])
    return texto


def carrega_pdf(caminho):
    loader = PyPDFLoader(caminho)
    lista_documentos = loader.load()
    documento = "\n\n".join([doc.page_content for doc in lista_documentos])
    return documento
