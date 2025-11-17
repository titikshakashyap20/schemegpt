from fastapi import FastAPI
from .ingest import router as ingest_router

app = FastAPI(title="SchemeGPT Backend")

@app.get("/")
def health():
    return {"status": "backend ok"}

# mount ingestion router
app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
