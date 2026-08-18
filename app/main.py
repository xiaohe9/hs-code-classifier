import time
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.agent.graph import classify

app = FastAPI(title="hs-code-classifier", version="0.3.0")


class ClassifyRequest(BaseModel):
    description: str = Field(min_length=2, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify")
def classify_api(req: ClassifyRequest):
    t0 = time.perf_counter()
    result = classify(req.description)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return result