from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

system = """You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation."""

answer_grader_system = """You are a grader assessing whether an
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""


class HallucinationGrader:
    _system = None
    _prompt = None
    _llm = None
    _hallucination_grader = None

    _answer_prompt = None
    _answer_grader = None

    def __init__(self, llm):
        self._system = system
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "question")
            ]
        )
        self._llm = llm
        self._answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "question: {question}\n\n answer: {generation} "),
            ]
        )
        self._answer_grader = self._answer_prompt | self._llm | JsonOutputParser()
        self._hallucination_grader = self._prompt | self._llm | JsonOutputParser()

    def grade_hallucination(self, state):
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self._hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            score = self._answer_grader.invoke({"question": question, "generation": generation})
            grade = score["score"]
            if grade == "yes":
                return "useful"
            else:
                return "not useful"
        else:
            return "not supported"
