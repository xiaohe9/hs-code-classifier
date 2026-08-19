"""三层防幻觉模块集成测试：输入硬拦截 / 过程交叉验证 / 输出置信度兜底"""
from app.guardrails.guards import input_guard, cross_validate, output_gate


def test_prohibited_input_blocked():
    """禁限品输入（象牙雕刻摆件）必须在第一层被拦截，不进 LLM"""
    r = input_guard("象牙雕刻摆件")
    assert r["blocked"] is True
    assert "象牙" in r["reason"]
    assert r["message"]  # 必须给出拦截提示信息


def test_low_confidence_needs_human_review():
    """低置信度结果必须被第三层标记 needs_human_review"""
    result = {"hs_code": "9405.42.00", "confidence": 0.3}
    r = output_gate(result, threshold=0.6)
    assert r["needs_human_review"] is True
    assert "置信度" in r["review_reason"]


def test_normal_classification_passes_all_layers():
    """正常商品归类必须依次通过全部三层，无需人工复核"""
    description = "圣诞节LED装饰灯串，低压24V"
    retrieved = [{"code": "9405.42.00"}, {"code": "8539.52.00"}]
    llm_result = {"hs_code": "9405.42.00", "confidence": 0.92}

    # 第一层：输入不拦截
    r1 = input_guard(description)
    assert r1["blocked"] is False

    # 第二层：LLM 输出编码在检索结果中，交叉验证通过
    r2 = cross_validate(llm_result, retrieved)
    assert r2["valid"] is True

    # 第三层：格式合法且置信度高于阈值，不触发人工复核
    r3 = output_gate(dict(llm_result), threshold=0.6)
    assert r3["needs_human_review"] is False
