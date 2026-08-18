import os

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
NUM_CTX = 8192  # 关键！Ollama 默认 2048/4096 会截断 RAG prompt，生产踩过的坑

# RAG
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "tax_rules"
VECTOR_TOP_K = 20        # 向量召回 top-20
BM25_TOP_K = 20          # BM25 召回 top-20
FINAL_TOP_N = 5          # 融合后取 top-5 进上下文

# 防幻觉
CONFIDENCE_THRESHOLD = 0.6   # 低于此值转人工复核