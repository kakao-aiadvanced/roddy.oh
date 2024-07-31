from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

system = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise"""


class AnswerGenerator:
    _system = None
    _prompt = None
    _llm = None
    _rag_chain = None

    def __init__(self, llm):
        self._system = system
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "question: {question}\n\n context: {context} "),
            ]
        )
        self._llm = llm
        self._rag_chain = self._prompt | self._llm | StrOutputParser()

    def generate_answer(self, state):
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self._rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


