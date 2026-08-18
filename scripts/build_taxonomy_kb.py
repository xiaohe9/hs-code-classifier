"""税则知识库构建：JSON -> embedding -> ChromaDB"""
"""税则知识库构建：JSON -> embedding -> ChromaDB"""
print("探针1: 脚本已启动")          # ← 加这行（docstring 之后、import 之前）
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import glob
import chromadb
import ollama
from app.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, OLLAMA_BASE_URL


def load_taxonomy(data_dir="data/taxonomy"):
    chunks, metadatas, ids = [], [], []
    for path in glob.glob(f"{data_dir}/*.json"):
        for item in json.load(open(path, encoding="utf-8")):
            # 检索文本 = 条文 + 章注 + 关键词 + 示例，拼在一起向量化
            text = (
                f"{item['text']} "
                f"{' '.join(item.get('notes', []))} "
                f"关键词:{'、'.join(item.get('keywords', []))} "
                f"示例:{'、'.join(item.get('examples', []))}"
            )
            chunks.append(text)
            ids.append(item["chunk_id"])
            metadatas.append({
                "code": item["code"],
                "chapter": item["chapter"],
                "heading": item["heading"],
                "code_prefix": item["code_prefix"],
                "text": item["text"],          # 原文单独存，输出依据链要用
                "notes": json.dumps(item.get("notes", []), ensure_ascii=False),
                "gir_reference": item.get("gir_reference", ""),
            })
    return chunks, metadatas, ids

def main():
    print("探针2: 进入 main", flush=True)
    client = ollama.Client(host=OLLAMA_BASE_URL)
    print("探针3: ollama client 创建成功", flush=True)
    db = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=chromadb.Settings(anonymized_telemetry=False),
)
    #db = chromadb.PersistentClient(path=CHROMA_DIR)
    print("探针4: chromadb 就绪", flush=True)
    try:
        db.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    coll = db.create_collection(COLLECTION_NAME)
    print("探针5: 集合创建成功", flush=True)

    chunks, metadatas, ids = load_taxonomy()
    print(f"探针6: 加载了 {len(chunks)} 条数据", flush=True)

    BATCH = 8
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        resp = client.embed(model=EMBED_MODEL, input=batch)
        coll.add(
            documents=batch,
            embeddings=resp["embeddings"],
            metadatas=metadatas[i:i + BATCH],
            ids=ids[i:i + BATCH],
        )
        print(f"已写入 {min(i + BATCH, len(chunks))}/{len(chunks)}", flush=True)
    print(f"知识库构建完成，共 {len(chunks)} 条，存储于 {CHROMA_DIR}", flush=True)


if __name__ == "__main__":
    main()