from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import shutil
import os

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ingested")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    filename: str = file.filename or ""

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    dest_path = os.path.join(UPLOAD_DIR, filename)

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"filename": filename, "path": dest_path}

# backend/backend_app/ingest.py
import os
import subprocess
from fastapi import BackgroundTasks

# existing router variable is used above; we append this endpoint
@router.post("/process_all")
async def process_all(background_tasks: BackgroundTasks):
    """
    Trigger background processing of all ingested PDFs (runs ai/extract.py using the ai venv).
    Returns immediately and runs the processing in background.
    """
    def _run():
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            # Path to the ai venv python executable (Windows)
            venv_python = os.path.join(repo_root, "ai", "venv", "Scripts", "python.exe")
            script = os.path.join(repo_root, "ai", "extract.py")
            # If the venv python doesn't exist, fallback to system python
            if not os.path.exists(venv_python):
                venv_python = "python"
            # Run the extractor script
            subprocess.run([venv_python, script], check=True)
        except Exception as e:
            # For production you would log this properly
            print("[process_all] processing failed:", e)

    background_tasks.add_task(_run)
    return {"status": "processing_started"}
