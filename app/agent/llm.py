"""LLM 调用封装：ollama(本地) / deepseek(云端) 双后端可切换"""
import json
import ollama
import httpx
from app.config import OLLAMA_BASE_URL, LLM_MODEL, NUM_CTX, LLM_BACKEND, DEEPSEEK_API_KEY

client = ollama.Client(host=OLLAMA_BASE_URL)


def _chat_ollama(system: str, user: str) -> str:
    resp = client.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        format="json",
        options={"num_ctx": NUM_CTX, "temperature": 0.1},
    )
    return resp["message"]["content"]


def _chat_deepseek(system: str, user: str) -> str:
    r = httpx.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def chat_json(system: str, user: str) -> dict:
    fn = _chat_deepseek if LLM_BACKEND == "deepseek" else _chat_ollama
    for attempt in range(2):
        try:
            return json.loads(fn(system, user))
        except json.JSONDecodeError:
            if attempt == 1:
                raise