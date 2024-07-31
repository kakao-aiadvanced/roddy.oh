def decide_to_generate(state):
    state["question"]
    state["documents"]
    web_search = state["web_search"]

    if web_search == "Yes":
        return "websearch"
    else:
        return "generate"