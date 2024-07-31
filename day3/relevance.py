from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

system = """You are a relevance checker for a search engine. Your task is to evaluate the relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """


class RelevanceChecker:
    _system = None
    _prompt = None
    _llm = None
    _retriever = None

    def __init__(self, llm):
        self._system = system
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._system),
                ("human", "question: {question}\n\n document: {document} "),
            ]
        )
        self._llm = llm
        self._retriever = self._prompt | self._llm | JsonOutputParser()

    def check_relevant(self, state):
        question = state["question"]
        document = state["document"]

        return self._retriever.invoke({"question": question, "document": document})
