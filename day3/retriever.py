class Retriever:
    _retriever = None

    def __init__(self, vector_store):
        self._retriever = vector_store.get_retriever()

    def retrieve(self, state):
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        question = state["question"]

        # Retrieval
        documents = self._retriever.invoke(question)
        return {"documents": documents, "question": question}