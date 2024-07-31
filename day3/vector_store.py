from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


class VectorStore:
    _openai_api_key = None
    _vector_store = None

    def __init__(self, openai_api_key):
        self._openai_api_key = openai_api_key

    def create_vector_store(self, documents):
        if self._vector_store is not None:
            return self._vector_store

        self._vector_store = Chroma.from_documents(
            documents=documents,
            collection_name="rag-chroma",
            embedding=OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=self._openai_api_key)
        )

        return self._vector_store

    def get_retriever(self):
        return self._vector_store.as_retriever()
