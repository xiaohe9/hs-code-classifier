"""LLM 调用封装：所有 ollama 调用收口在这，方便换底座"""
import json
import ollama
from app.config import OLLAMA_BASE_URL, LLM_MODEL, NUM_CTX

client = ollama.Client(host=OLLAMA_BASE_URL)

def chat_json(system: str, user: str) -> dict:
    """强制 JSON 输出，解析失败重试一次"""
    for attempt in range(2):
        resp = client.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format="json",
            options={"num_ctx": NUM_CTX, "temperature": 0.1},  # 归类场景低温
        )
        try:
            return json.loads(resp["message"]["content"])
        except json.JSONDecodeError:
            if attempt == 1:
                raise