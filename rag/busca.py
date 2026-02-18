from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_PATH = os.path.join(BASE_DIR, "base_vetorial")


def buscar_contexto(pergunta, k=4):

    if not os.path.exists(VECTOR_PATH):
        return ""

    try:
        embeddings = OpenAIEmbeddings()

        db = FAISS.load_local(
            VECTOR_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = db.similarity_search(pergunta, k=k)

        return "\n\n".join(doc.page_content for doc in docs)

    except:
        return ""
