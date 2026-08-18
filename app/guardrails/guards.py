"""三层防幻觉：输入硬拦截 / 过程交叉验证 / 输出置信度兜底"""
import re

# ── 第一层：输入硬规则（禁限品目录示例，真实业务接海关禁限目录）──
PROHIBITED_PATTERNS = [
    r"枪支|弹药|爆炸物",
    r"毒品|冰毒|海洛因|大麻",
    r"管制刀具|弩",
    r"假币|伪造证件",
    r"象牙|犀牛角|濒危动植物",
]

def input_guard(description: str) -> dict:
    """命中禁限品直接拦截，不进 LLM"""
    for pat in PROHIBITED_PATTERNS:
        if re.search(pat, description):
            return {"blocked": True, "reason": f"命中禁限品规则: {pat}",
                    "message": "该商品疑似属于禁止/限制进出境物品，请直接咨询持证报关员，系统不提供归类建议。"}
    if len(description.strip()) < 4:
        return {"blocked": True, "reason": "描述过短",
                "message": "商品描述信息过少，请补充材质、用途、功能等信息。"}
    return {"blocked": False}


# ── 第二层：过程交叉验证（结论必须映射回检索到的条文）──
def cross_validate(llm_result: dict, retrieved: list[dict]) -> dict:
    """LLM 输出的编码必须在检索结果里，否则作废"""
    valid_codes = {r["code"] for r in retrieved}
    code = llm_result.get("hs_code", "")
    if code not in valid_codes:
        return {"valid": False,
                "reason": f"LLM 输出编码 {code} 不在检索结果 {valid_codes} 中，判定为幻觉，作废"}
    return {"valid": True}


# ── 第三层：输出兜底（置信度阈值 + 编码格式校验）──
CODE_PATTERN = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")

def output_gate(result: dict, threshold: float) -> dict:
    if not CODE_PATTERN.match(result.get("hs_code", "")):
        result["needs_human_review"] = True
        result["review_reason"] = "编码格式校验失败"
        return result
    if result.get("confidence", 0) < threshold:
        result["needs_human_review"] = True
        result["review_reason"] = (
            f"置信度 {result['confidence']:.2f} 低于阈值 {threshold}，"
            "建议持证报关员复核候选编码"
        )
    else:
        result["needs_human_review"] = False
    return result