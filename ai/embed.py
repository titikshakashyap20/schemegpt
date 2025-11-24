# ai/embed.py
import os
import json
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(REPO_ROOT, "backend", "data", "processed")
VECTOR_DIR = os.path.join(REPO_ROOT, "backend", "data", "vectorstore")

os.makedirs(VECTOR_DIR, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "schemes"


def load_chunks():
    docs = []
    for fname in os.listdir(PROCESSED_DIR):
        if fname.endswith("_chunks.json"):
            base = fname.replace("_chunks.json", "")
            path = os.path.join(PROCESSED_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            for i, chunk in enumerate(chunks):
                doc_id = f"{base}__chunk__{i}"
                docs.append((doc_id, chunk, {"source": base, "chunk_index": i}))
    return docs


def reset_collection(client):
    """Safely drop and recreate the collection."""
    collections = client.list_collections()

    if any(c.name == COLLECTION_NAME for c in collections):
        client.delete_collection(COLLECTION_NAME)

    return client.create_collection(COLLECTION_NAME)


def main():
    docs = load_chunks()
    if not docs:
        print("No chunks found.")
        return

    print(f"Found {len(docs)} chunks to embed.")
    model = SentenceTransformer(MODEL_NAME)

    client = PersistentClient(path=VECTOR_DIR)

    # ← THIS FIXES YOUR ERROR
    collection = reset_collection(client)

    ids = [d[0] for d in docs]
    texts = [d[1] for d in docs]
    metadata = [d[2] for d in docs]

    print("Creating embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadata,
        embeddings=embeddings.tolist()
    )

    print("Embedding complete. Saved at:", VECTOR_DIR)


if __name__ == "__main__":
    main()
