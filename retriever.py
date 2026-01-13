from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def build_bm25(chunks):
    tokenized = [c.split() for c in chunks]
    return BM25Okapi(tokenized)
