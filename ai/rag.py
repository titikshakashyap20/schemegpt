# ai/rag.py
import os
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from openai import OpenAI

# ----------- CONFIG -----------

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment.")

client_llm = OpenAI(api_key=OPENAI_KEY)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VECTOR_DIR = os.path.join(REPO_ROOT, "backend", "data", "vectorstore")

# Vector DB
chroma = PersistentClient(path=VECTOR_DIR)
collection = chroma.get_collection("schemes")

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------- SCHEME DETECTION -----------

SCHEME_HINTS: dict[str, list[str]] = {
    "pmjdy": [
        "pmjdy",
        "jan dhan",
        "pradhan mantri jan dhan",
        "pradhan mantri jan dhan yojana",
    ],
    "nsp": [
        "nsp",
        "scholarship",
        "national scholarship",
        "national scholarship portal",
    ],
    "ayushman": [
        "ayushman",
        "pmjay",
        "pm-jay",
        "pradhan mantri jan arogya",
        "ayushman bharat",
    ],
    "pmay-g": [
        "pmay-g",
        "pmay g",
        "rural housing",
        "gramin awas",
        "pradhan mantri awas yojana gramin",
    ],
    "pmay-u": [
        "pmay-u",
        "pmay u",
        "urban housing",
        "pradhan mantri awas yojana urban",
    ],
    "mudra": [
        "mudra",
        "mudra loan",
        "pradhan mantri mudra yojana",
        "pmmy",
    ],
}

SCHEME_DISPLAY_NAMES: dict[str, str] = {
    "pmjdy": "Pradhan Mantri Jan Dhan Yojana (PMJDY)",
    "nsp": "National Scholarship Portal (NSP)",
    "ayushman": "Ayushman Bharat – PM-JAY",
    "pmay-g": "Pradhan Mantri Awas Yojana – Gramin (PMAY-G)",
    "pmay-u": "Pradhan Mantri Awas Yojana – Urban (PMAY-U)",
    "mudra": "Pradhan Mantri Mudra Yojana (MUDRA)",
}


def detect_scheme(question: str) -> Optional[str]:
    """Return canonical scheme key like 'pmjdy', 'nsp', etc. if detected."""
    q = question.lower()
    for scheme_key, keywords in SCHEME_HINTS.items():
        for kw in keywords:
            if kw in q:
                return scheme_key
    return None


def _normalize_source(src: str) -> str:
    """Normalize metadata['source'] to improve scheme matching."""
    s = src.lower()
    s = s.replace(".pdf", "").replace("_", "-").replace(" ", "-")
    return s


def _scheme_from_source(source: str) -> Optional[str]:
    """Best-effort map from file/source name to scheme_key."""
    norm = _normalize_source(source)
    for scheme_key in SCHEME_HINTS.keys():
        if scheme_key in norm:
            return scheme_key
    return None


# ----------- RETRIEVAL -----------

def retrieve_chunks(query: str, k: int = 20) -> Dict[str, Any]:
    """Retrieve top-k chunks with scheme-aware query enrichment and distances."""
    scheme_key = detect_scheme(query)
    enhanced_query = query

    # Enrich query to pull closer to scheme-specific content
    if scheme_key == "pmjdy":
        enhanced_query = (
            f"{query} PMJDY Pradhan Mantri Jan Dhan Yojana bank account "
            f"life cover eligibility rules"
        )
    elif scheme_key == "nsp":
        enhanced_query = (
            f"{query} NSP National Scholarship Portal scholarship eligibility "
            f"income criteria student benefits"
        )
    elif scheme_key == "ayushman":
        enhanced_query = (
            f"{query} Ayushman Bharat PMJAY Pradhan Mantri Jan Arogya Yojana "
            f"health insurance eligibility coverage"
        )
    elif scheme_key == "pmay-g":
        enhanced_query = (
            f"{query} PMAY-G Pradhan Mantri Awas Yojana Gramin rural housing "
            f"eligibility SECC 2011 criteria"
        )
    elif scheme_key == "pmay-u":
        enhanced_query = (
            f"{query} PMAY-U Pradhan Mantri Awas Yojana Urban urban housing "
            f"subsidy eligibility"
        )
    elif scheme_key == "mudra":
        enhanced_query = (
            f"{query} MUDRA loan Pradhan Mantri Mudra Yojana PMMY loan "
            f"eligibility shishu kishore tarun"
        )

    emb = embed_model.encode([enhanced_query], convert_to_numpy=True)[0].tolist()

    results = collection.query(
        query_embeddings=[emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def filter_by_scheme(
    docs: List[str],
    metas: List[Dict[str, Any]],
    dists: List[float],
    scheme_key: Optional[str],
) -> tuple[List[str], List[Dict[str, Any]], List[float]]:
    """If scheme is detected, keep only chunks whose source looks like that scheme."""
    if not scheme_key:
        return docs, metas, dists

    filtered_docs: List[str] = []
    filtered_metas: List[Dict[str, Any]] = []
    filtered_dists: List[float] = []

    for doc, meta, dist in zip(docs, metas, dists):
        src = _normalize_source(str(meta.get("source", "")))
        if scheme_key in src:
            filtered_docs.append(doc)
            filtered_metas.append(meta)
            filtered_dists.append(dist)

    # If we found scheme-specific chunks, use only them
    if filtered_docs:
        return filtered_docs, filtered_metas, filtered_dists

    # Otherwise fall back to original results
    return docs, metas, dists


# ----------- SCORING & CONTEXT -----------

def distance_to_similarity(d: float) -> float:
    """Map distance to a 0–1 similarity score (monotonic)."""
    # Chroma default distances are usually cosine distances in [0, 2].
    # This is a simple, monotonic mapping: higher = more similar.
    return 1.0 / (1.0 + d)


def compute_confidence(distances: List[float]) -> float:
    """Compute an overall confidence score (0–1) from top distances."""
    if not distances:
        return 0.0
    sims = [distance_to_similarity(d) for d in distances]
    top_sims = sorted(sims, reverse=True)[:3]
    avg_top = sum(top_sims) / len(top_sims)
    # Clamp to [0, 1]
    return max(0.0, min(1.0, avg_top))


def build_context(docs: List[str], metas: List[Dict[str, Any]]) -> str:
    """Build a readable context block for the LLM."""
    parts: List[str] = []
    for doc, meta in zip(docs, metas):
        source = meta.get("source", "unknown")
        idx = meta.get("chunk_index", "?")
        snippet = doc.strip()

        parts.append(
            f"Source: {source} (chunk {idx})\n"
            f"{snippet}\n"
        )
    return "\n---\n".join(parts)


# ----------- MAIN RAG ANSWER -----------

def answer_with_rag(question: str) -> Dict[str, Any]:
    """Main RAG answer generator with conversational tone and confidence."""
    # Step 1 — retrieve
    retrieved = retrieve_chunks(question, k=20)
    docs = retrieved["documents"][0]
    metas = retrieved["metadatas"][0]
    dists = retrieved.get("distances", [[ ]] )[0] or [ ]

    # Step 2 — scheme-aware filtering
    detected_scheme_key = detect_scheme(question)
    docs, metas, dists = filter_by_scheme(docs, metas, dists, detected_scheme_key)

    # Step 3 — add similarity scores into metadata
    similarities = [distance_to_similarity(d) for d in dists]
    for meta, sim in zip(metas, similarities):
        meta["similarity_score"] = round(sim, 4)

    # Step 4 — compute overall confidence
    confidence = compute_confidence(dists)

    # Step 5 — build context for LLM
    context = build_context(docs, metas)

    # For display, prefer scheme from detection; fall back to source-derived
    display_scheme_name: Optional[str] = None
    if detected_scheme_key and detected_scheme_key in SCHEME_DISPLAY_NAMES:
        display_scheme_name = SCHEME_DISPLAY_NAMES[detected_scheme_key]
    else:
        # Try infer from top source
        if metas:
            src_scheme_key = _scheme_from_source(str(metas[0].get("source", "")))
            if src_scheme_key and src_scheme_key in SCHEME_DISPLAY_NAMES:
                display_scheme_name = SCHEME_DISPLAY_NAMES[src_scheme_key]

    # Step 6 — conversational, structured prompt
    prompt = f"""
You are SchemeGPT, a friendly assistant that explains Indian government schemes.

USER QUESTION:
{question}

DETECTED SCHEME (if any): {display_scheme_name or "Unknown / Mixed"}

You are given CONTEXT from official scheme PDFs. Follow these rules:

1. Use ONLY the facts from the CONTEXT. Do NOT guess or add outside knowledge.
2. Answer in **simple, conversational English**, like you're explaining to a friend.
3. Start with a short 2–3 line summary.
4. Then give clear bullet points for key eligibility / benefits / rules.
5. If the answer is not present in the context, reply exactly:
   "The information is not available in the provided documents."

CONTEXT:
{context}

Now write a helpful, conversational answer based ONLY on the context:
"""

    response = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content.strip()

    return {
        "question": question,
        "answer": answer,
        "sources": metas,
        "detected_scheme": detected_scheme_key,
        "confidence": round(confidence, 4),
    }


if __name__ == "__main__":
    print(answer_with_rag("What is the eligibility for PMJDY?"))
