"""混合检索：ChromaDB 向量召回 + BM25 精确匹配，加权融合"""
import json
import jieba
import chromadb
import ollama
from rank_bm25 import BM25Okapi
from app.config import (
    CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, OLLAMA_BASE_URL,
    VECTOR_TOP_K, BM25_TOP_K, FINAL_TOP_N,
)


def _tokenize(text: str) -> list[str]:
    """中文分词：jieba 为主，保留编码类 token 的整体性"""
    return [t for t in jieba.lcut(text) if t.strip()]


class HybridRetriever:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        db = chromadb.PersistentClient(path=CHROMA_DIR)
        self.coll = db.get_collection(COLLECTION_NAME)
        # 启动时把全量文档拉出建 BM25 索引（数据量小，内存索引足够；
        # 百万级才需要 Elasticsearch，面试可主动讲这个权衡）
        all_data = self.coll.get(include=["documents", "metadatas"])
        self.docs = all_data["documents"]
        self.metas = all_data["metadatas"]
        self.bm25 = BM25Okapi([_tokenize(d) for d in self.docs])

    def search(self, query: str, mode: str = "hybrid") -> list[dict]:
        """mode: vector(纯向量) / bm25(纯关键词) / hybrid(融合)"""
        scores: dict[str, float] = {}
        info: dict[str, dict] = {}
        all_ids = self.coll.get()["ids"]

        # 向量分支
        if mode in ("vector", "hybrid"):
            emb = self.client.embed(model=EMBED_MODEL, input=[query])["embeddings"]
            vres = self.coll.query(query_embeddings=emb, n_results=VECTOR_TOP_K)
            for rank, (doc_id, meta) in enumerate(
                zip(vres["ids"][0], vres["metadatas"][0])
            ):
                scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rank + 1)
                info[doc_id] = meta

        # BM25 分支
        if mode in ("bm25", "hybrid"):
            bm_scores = self.bm25.get_scores(_tokenize(query))
            top_idx = sorted(range(len(bm_scores)),
                             key=lambda i: bm_scores[i], reverse=True)[:BM25_TOP_K]
            for rank, idx in enumerate(top_idx):
                if bm_scores[idx] <= 0:
                    continue
                doc_id = all_ids[idx]
                scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rank + 1)
                info[doc_id] = self.metas[idx]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:FINAL_TOP_N]
        return [
            {"chunk_id": cid, "score": round(s, 4), "code": info[cid]["code"],
             "text": info[cid]["text"],
             "notes": json.loads(info[cid].get("notes") or "[]"),
             "gir_reference": info[cid].get("gir_reference", "")}
            for cid, s in ranked
        ]


if __name__ == "__main__":
    r = HybridRetriever()
    for hit in r.search("圣诞节LED灯串"):
        print(hit["code"], hit["score"], hit["text"][:30])