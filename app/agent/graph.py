from langgraph.graph import StateGraph, END
from app.agent.state import ClassifyState
from app.agent.nodes import guard_node, check_info_node, classify_node


def _after_guard(state):
    return END if state.get("blocked") else "check_info"

def _after_check(state):
    return "classify" if state.get("info_sufficient") else END  # 追问则本轮结束


def build_graph():
    g = StateGraph(ClassifyState)
    g.add_node("guard", guard_node)
    g.add_node("check_info", check_info_node)
    g.add_node("classify", classify_node)
    g.set_entry_point("guard")
    g.add_conditional_edges("guard", _after_guard)
    g.add_conditional_edges("check_info", _after_check)
    g.add_edge("classify", END)
    return g.compile()


graph = build_graph()


def classify(description: str) -> dict:
    state = graph.invoke({"description": description, "trace": []})
    if state.get("blocked"):
        return {"status": "blocked", "message": state["block_message"],
                "trace": state["trace"]}
    if not state.get("info_sufficient"):
        return {"status": "need_clarify",
                "questions": state["clarify_questions"],
                "candidates": [h["code"] for h in state["retrieved"][:3]],
                "trace": state["trace"]}
    out = state["final"]
    out["status"] = "ok"
    out["trace"] = state["trace"]
    return out