from langgraph.graph import END, StateGraph

from graph import GraphState
from retriever import Retriever
from answer import AnswerGenerator
from url_loader import UrlLoader
from vector_store import VectorStore
from relevance import RelevanceChecker
from hallucination import HallucinationGrader
from web_search import WebSearch
from decide_to_generate import decide_to_generate

from langchain_openai import ChatOpenAI
from pprint import pprint

# TODO: remove this

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=API_KEY)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

urlLoader = UrlLoader()
documents = urlLoader.load_docs_from_urls(urls)

vectorStore = VectorStore(API_KEY)
vectorStore.create_vector_store(documents)
retriever = vectorStore.get_retriever()

retriever = Retriever(vectorStore)
web_searcher = WebSearch(TAVILY_API_KEY)
relevant_checker = RelevanceChecker(llm)
answer_generator = AnswerGenerator(llm)
hallucination_grader = HallucinationGrader(llm)



# Build graph
workflow = StateGraph(GraphState)

"""
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
"""

workflow.add_node("retriever", retriever.retrieve)
workflow.add_node("grade_documents", relevant_checker.check_relevant)
workflow.add_node("answer_generator", answer_generator.generate_answer)
workflow.add_node("websearch", web_searcher.search)
workflow.add_node("hallucination_grader", hallucination_grader.grade_hallucination)


workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate,
                               {"yes": "answer_generator", "no": "websearch"})
workflow.add_edge("answer_generator", "hallucination_grader")
workflow.add_conditional_edges("hallucination_grader", hallucination_grader.grade_hallucination,
                               {"useful": END, "not useful": "websearch", "not supported": "websearch"})
workflow.add_edge("websearch", "retriever")


app = workflow.compile()
inputs = {"question": "Where does Messi play right now?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")

pprint(value["generation"])


