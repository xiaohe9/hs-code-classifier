"""MCP Client 封装：Agent 通过 SSE 调用税则工具"""
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

MCP_URL = "http://localhost:8765/sse"


async def _call_tool(name: str, args: dict):
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, args)
            sc = result.structuredContent
            if sc is not None:
                # FastMCP 会把 list 返回值包成 {"result": [...]}
                if isinstance(sc, dict) and "result" in sc:
                    return sc["result"]
                return sc
            import json
            return json.loads(result.content[0].text)


def call_tool(name: str, args: dict):
    """同步包装：LangGraph 节点是同步函数"""
    return asyncio.run(_call_tool(name, args))