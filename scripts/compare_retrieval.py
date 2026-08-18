import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.retriever import HybridRetriever

# (查询, 期望命中的编码前缀) —— 按你数据的实际情况调整
QUERIES = [
    ("圣诞节装饰灯串", "9405"),                    # 语义型
    ("给小孩玩的电动遥控车", "9503.00.83"),         # 语义型
    ("9503.00.83 是什么商品", "9503.00.83"),       # 编码型
    ("8544开头的线缆", "8544"),                     # 编码型
    ("LED strip 灯串", "9405.42"),                 # 中英混合
]

r = HybridRetriever()
hit = {"vector": 0, "bm25": 0, "hybrid": 0}

hit1 = {"vector": 0, "bm25": 0, "hybrid": 0}
hit3 = {"vector": 0, "bm25": 0, "hybrid": 0}

for q, expect in QUERIES:
    print(f"\n{'='*60}\n查询: {q}   期望: {expect}")
    for mode in ("vector", "bm25", "hybrid"):
        hits = r.search(q, mode=mode)
        codes = [h["code"] for h in hits[:3]]
        ok1 = bool(codes) and codes[0].startswith(expect)
        ok3 = any(c.startswith(expect) for c in codes)
        hit1[mode] += ok1
        hit3[mode] += ok3
        print(f"  {mode:8s} top3={codes}  top1{'✅' if ok1 else '❌'} top3{'✅' if ok3 else '❌'}")

print(f"\n{'='*60}\n命中率统计({len(QUERIES)} 条查询):")
for mode in hit1:
    print(f"  {mode:8s}: Top-1 {hit1[mode]}/{len(QUERIES)}   Top-3 {hit3[mode]}/{len(QUERIES)}")