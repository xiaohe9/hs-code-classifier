"""评测：Top-1 / Top-3 / 品目级准确率 / 拦截率 / 延迟
运行前确保: 1) MCP Server 已启动  2) 需要云端速度时设置 $env:LLM_BACKEND="deepseek"
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
from app.agent.graph import classify
from app.config import LLM_BACKEND, LLM_MODEL


def main():
    cases = json.load(open("tests/eval/test_cases.json", encoding="utf-8"))
    top1 = top3 = heading_ok = blocked_ok = clarify_ok = 0
    n_normal = n_blocked = n_clarify = 0
    latencies = []
    bad_cases = []

    for i, c in enumerate(cases, 1):
        t0 = time.perf_counter()
        r = classify(c["description"])
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        verdict = ""
        if c["type"] == "adversarial_blocked":
            n_blocked += 1
            ok = r.get("status") == "blocked"
            blocked_ok += ok
            verdict = "拦截" + ("✅" if ok else "❌")
        elif c["type"] == "adversarial_clarify":
            n_clarify += 1
            ok = r.get("status") == "need_clarify" or (r.get("needs_human_review") is True)
            clarify_ok += ok
            verdict = "追问" + ("✅" if ok else "❌")
        else:
            n_normal += 1
            if r.get("status") != "ok":
                verdict = f"❌ 非正常返回({r.get('status')})"
                ok = False
            else:
                got, exp = r["hs_code"], c["expected_code"]
                cands = [got] + r.get("alternatives", [])
                is_top1 = got == exp
                is_top3 = exp in cands
                is_heading = got[:4] == exp[:4]
                top1 += is_top1
                top3 += is_top3
                heading_ok += is_heading
                ok = is_top1
                verdict = f"{'✅' if is_top1 else '❌'} 预测{got} 期望{exp}"
            if not ok:
                bad_cases.append({
                    "id": c["id"],
                    "description": c["description"],
                    "expected": c.get("expected_code"),
                    "got": r.get("hs_code") or r.get("status"),
                })

        print(f"[{i}/{len(cases)}] {c['type']}: {verdict}  {ms:.0f}ms", flush=True)

    latencies.sort()
    report = {
        "llm_backend": LLM_BACKEND if LLM_BACKEND == "deepseek" else f"ollama/{LLM_MODEL}",
        "样本数": len(cases),
        "Top-1准确率": f"{top1}/{n_normal} = {top1 / max(n_normal, 1):.1%}",
        "Top-3命中率": f"{top3}/{n_normal} = {top3 / max(n_normal, 1):.1%}",
        "品目级准确率": f"{heading_ok}/{n_normal} = {heading_ok / max(n_normal, 1):.1%}",
        "拦截正确率": f"{blocked_ok}/{n_blocked}",
        "追问正确率": f"{clarify_ok}/{n_clarify}",
        "平均延迟ms": round(sum(latencies) / len(latencies)),
        "P95延迟ms": round(latencies[int(len(latencies) * 0.95)]),
        "bad_cases": bad_cases,
    }
    print("\n===== 评测报告 =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    json.dump(report, open("tests/eval/latest_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()