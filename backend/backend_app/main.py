import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make project root importable (so `ai` works)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from backend_app.ingest import router as ingest_router  # type: ignore
from ai.rag import answer_with_rag  # type: ignore


# ----------- FastAPI app setup -----------

app = FastAPI(title="SchemeGPT Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------- Models -----------

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    detected_scheme: str | None = None


# ----------- Routes -----------

@app.get("/")
def health():
    return {"status": "backend ok"}


# Ingestion routes
app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])


@app.post("/query", response_model=QueryResponse)
def query_api(payload: QueryRequest):
    result = answer_with_rag(payload.question)
    return result


