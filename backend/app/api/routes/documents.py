"""Document endpoints: upload / list / delete business docs, and search them."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...config import get_settings
from ...knowledge import definitions, get_document_store
from ...knowledge.answer import answer
from ...knowledge.parsing import UnsupportedDocument
from ...knowledge.retrieval import MODES, search

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _enabled() -> None:
    if not get_settings().enable_docs:
        raise HTTPException(status_code=404, detail="Document RAG is disabled (ENABLE_DOCS=false)")


@router.post("")
def upload_documents(files: list[UploadFile] = File(...)):
    """Upload PDF / Markdown / TXT / DOCX documents. Re-uploading a document
    with the same name replaces it; unchanged chunks are not re-embedded."""
    _enabled()
    store = get_document_store()
    imported, errors = [], []
    for upload in files:
        try:
            imported.append(store.add(upload.filename or "document.txt", upload.file.read()))
        except (UnsupportedDocument, ValueError) as exc:
            errors.append({"file": upload.filename, "error": str(exc)})
    definitions.reset_cache()
    status = 207 if errors and imported else (400 if errors else 200)
    return JSONResponse({"imported": imported, "errors": errors, "count": len(imported)}, status_code=status)


@router.get("")
def list_documents():
    _enabled()
    docs = get_document_store().list()
    return {"documents": [d.to_api() for d in docs], "count": len(docs),
            "definitions": [d.to_api() for d in definitions.all_definitions()]}


@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    _enabled()
    if not get_document_store().delete(doc_id):
        raise HTTPException(status_code=404, detail=f"Unknown document id: {doc_id}")
    definitions.reset_cache()
    return {"deleted": doc_id}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=20)
    mode: str = Field("hybrid_rerank", description=f"one of {', '.join(MODES)}")
    answer: bool = Field(False, description="also generate a cited answer from the hits")


@router.post("/search")
def search_documents(req: SearchRequest):
    """Retrieve passages (with every retriever's rank and the rerank score) and
    optionally a cited answer. Useful for inspecting the retrieval pipeline."""
    _enabled()
    if req.mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {MODES}")
    result = search(req.query, k=req.k, mode=req.mode)
    payload = result.to_api()
    if req.answer:
        from ...core.llm import get_llm

        payload["answer"] = answer(req.query, result, llm=get_llm())
    return payload
