from app.guardrails.guards import input_guard, cross_validate, output_gate

def test_prohibited_blocked():
    r = input_guard("象牙雕刻摆件")
    assert r["blocked"] is True

def test_normal_pass():
    assert input_guard("圣诞节LED装饰灯串，低压24V")["blocked"] is False

def test_cross_validate_rejects_hallucination():
    retrieved = [{"code": "9405.42.00"}]
    r = cross_validate({"hs_code": "9999.99.99"}, retrieved)
    assert r["valid"] is False

def test_output_gate_low_confidence():
    r = output_gate({"hs_code": "9405.42.00", "confidence": 0.3}, 0.6)
    assert r["needs_human_review"] is True

def test_output_gate_bad_format():
    r = output_gate({"hs_code": "9405", "confidence": 0.9}, 0.6)
    assert r["needs_human_review"] is True