from tavily import TavilyClient
from langchain_core.documents import Document

class WebSearch:
    _api_key = None
    _client = None

    def __init__(self, api_key):
        self._api_key = api_key
        self._client = TavilyClient(api_key)

    def search(self, state):
        question = state["question"]
        documents = state["documents"]

        docs = self._client.search(query=question)['results']

        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {"documents": documents, "question": question}
