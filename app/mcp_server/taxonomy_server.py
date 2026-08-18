"""税则检索 MCP Server：把检索能力封装成标准 MCP Tool"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcp.server.fastmcp import FastMCP
from app.rag.retriever import HybridRetriever

mcp = FastMCP("hs-code-taxonomy", host="0.0.0.0", port=8765)

_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


@mcp.tool()
def search_taxonomy(query: str, top_n: int = 5) -> list[dict]:
    """检索海关进出口税则条文。输入商品描述（支持中英文），返回最相关的税则条目，
    包含编码、条文原文、类章注和归类总规则依据。归类前必须先调用此工具获取条文，
    禁止凭空生成编码。"""
    return get_retriever().search(query)[:top_n]


@mcp.tool()
def validate_hs_code(code: str) -> dict:
    """校验 HS 编码是否存在于税则库中。LLM 输出的编码必须经此工具验证后方可返回。"""
    hits = get_retriever().search(code, top_n=20)
    exists = any(h["code"] == code for h in hits)
    return {"code": code, "exists": exists}


if __name__ == "__main__":
    mcp.run(transport="sse")