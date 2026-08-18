import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb, ollama
from app.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, OLLAMA_BASE_URL

client = ollama.Client(host=OLLAMA_BASE_URL)
db = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=chromadb.Settings(anonymized_telemetry=False),
)
#db = chromadb.PersistentClient(path=CHROMA_DIR)
coll = db.get_collection(COLLECTION_NAME)

q = "圣诞节用的LED装饰灯串"
emb = client.embed(model=EMBED_MODEL, input=[q])["embeddings"]
res = coll.query(query_embeddings=emb, n_results=3)
for code, dist in zip(res["metadatas"][0], res["distances"][0]):
    print(f"{code['code']}  距离={dist:.4f}  {code['text'][:40]}...")
# 预期：9405.42.00 排第一