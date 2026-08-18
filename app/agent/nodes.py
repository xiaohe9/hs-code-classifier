from app.agent.llm import chat_json
from app.guardrails.guards import input_guard, cross_validate, output_gate
from app.rag.retriever import HybridRetriever
from app.config import CONFIDENCE_THRESHOLD
from app.mcp_server.client import call_tool

_retriever = None
def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def guard_node(state):
    """节点1：输入护栏"""
    r = input_guard(state["description"])
    trace = state.get("trace", []) + [f"输入护栏: {'拦截' if r['blocked'] else '通过'}"]
    if r["blocked"]:
        return {"blocked": True, "block_message": r["message"], "trace": trace}
    return {"blocked": False, "trace": trace}


def check_info_node(state):
    """节点2：信息完备性检查 + 检索"""
    hits = call_tool("search_taxonomy", {"query": state["description"], "top_n": 5})
    #hits = get_retriever().search(state["description"])
    candidates = "\n".join(
        f"- {h['code']}: {h['text']}" for h in hits[:3]
    )
    r = chat_json(
        system=(
            "你是海关商品归类专家。根据商品描述和候选税则条文，判断现有信息是否足以完成归类。"
            "如果区分候选编码还缺关键属性（如材质、是否电动、用途、适用人群），列出最多3个追问问题。"
            '输出JSON: {"sufficient": true/false, "questions": ["..."]}'
        ),
        user=f"商品描述：{state['description']}\n\n候选条文：\n{candidates}",
    )
    trace = state.get("trace", []) + [
        f"信息检查: {'充分' if r.get('sufficient') else '需追问'}"
    ]
    return {
        "info_sufficient": bool(r.get("sufficient")),
        "clarify_questions": r.get("questions", []),
        "retrieved": hits,
        "trace": trace,
    }


def classify_node(state):
    """节点3：LLM 归类 + 三层防幻觉过滤"""
    rules_text = "\n\n".join(
        f"[{h['chunk_id']}] {h['code']}\n{h['text']}\n"
        f"章注: {';'.join(h['notes']) if h['notes'] else '无'}\n依据: {h['gir_reference']}"
        for h in state["retrieved"]
    )
    r = chat_json(
        system=(
            "你是海关商品归类专家。只能从给定条文范围内选择编码，禁止编造条文之外的编码。"
            "必须引用支持你结论的条文 chunk_id。"
            '输出JSON: {"hs_code":"xxxx.xx.xx","confidence":0.0-1.0,'
            '"basis_chunk_ids":["..."],"reasoning":"...","alternatives":["..."]}'
        ),
        user=f"商品描述：{state['description']}\n\n可用条文：\n{rules_text}",
    )
    trace = state.get("trace", []) + [f"LLM归类: {r.get('hs_code')} 置信度{r.get('confidence')}"]

    # 第二层：交叉验证
    cv = cross_validate(r, state["retrieved"])
    if not cv["valid"]:
        trace.append(f"交叉验证失败: {cv['reason']}")
        # 降级：取检索第一名作为候选，强制人工复核
        r = {"hs_code": state["retrieved"][0]["code"], "confidence": 0.0,
             "reasoning": "LLM输出未通过交叉验证，降级为检索首位候选",
             "alternatives": [h["code"] for h in state["retrieved"][1:3]]}

    # 第三层：输出兜底
    basis = [
        {"code": h["code"], "text": h["text"], "notes": h["notes"],
         "gir_reference": h["gir_reference"]}
        for h in state["retrieved"]
        if h["chunk_id"] in r.get("basis_chunk_ids", [])
    ] or [{"code": h["code"], "text": h["text"]} for h in state["retrieved"][:1]]

    final = output_gate({
        "hs_code": r["hs_code"],
        "confidence": round(float(r.get("confidence", 0)), 2),
        "basis": basis,
        "reasoning": r.get("reasoning", ""),
        "alternatives": r.get("alternatives", [])[:2],
    }, CONFIDENCE_THRESHOLD)
    return {"llm_result": r, "final": final, "trace": trace}