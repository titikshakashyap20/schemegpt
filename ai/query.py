# ai/query.py
import os
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VECTOR_DIR = os.path.join(REPO_ROOT, "backend", "data", "vectorstore")

client = PersistentClient(path=VECTOR_DIR)
collection = client.get_collection("schemes")

model = SentenceTransformer("all-MiniLM-L6-v2")

def query(text, k=5):
    emb = model.encode([text], convert_to_numpy=True)[0].tolist()
    results = collection.query(
        query_embeddings=[emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    return results

if __name__ == "__main__":
    q = "Eligibility criteria for Ayushman Bharat"
    res = query(q)
    print("\n=== QUERY RESULTS ===\n")
    for i in range(len(res["documents"][0])):
        print(f"Result {i+1}:")
        print("Distance:", res["distances"][0][i])
        print("Metadata:", res["metadatas"][0][i])
        print("Snippet:", res["documents"][0][i][:300], "...")
        print("---------------------")
