import json
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import classify, graph

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="hs-code-classifier", version="1.1.0")


class ClassifyRequest(BaseModel):
    description: str = Field(min_length=2, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/demo")
def demo_page():
    return FileResponse(BASE_DIR / "docs" / "demo.html")


@app.post("/classify")
def classify_api(req: ClassifyRequest):
    t0 = time.perf_counter()
    result = classify(req.description)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return result


@app.post("/classify/stream")
async def classify_stream(req: ClassifyRequest):
    async def event_gen():
        try:
            # stream_mode="values"：每执行完一个节点，吐出一次完整状态
            async for state in graph.astream(
                {"description": req.description, "trace": []},
                stream_mode="values",
            ):
                payload = {"state": state}
                yield f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")