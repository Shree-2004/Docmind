import os
import hashlib
import json
import requests
from pathlib import Path
from typing import List, Dict, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
)

# ── Constants ────────────────────────────────────────────────────────────────
CACHE_DIR = Path(".cache/ingestion")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CHILD_CHUNK_SIZE     = 400
CHILD_CHUNK_OVERLAP  = 60
PARENT_CHUNK_SIZE    = 1500
PARENT_CHUNK_OVERLAP = 150

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt"}

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "mistral"

SUMMARY_PROMPT = """\
Read the following document excerpt and write a 2-3 sentence summary describing:
1. What the document is about
2. The main topics it covers

Be concise and factual. Do not add any preamble.

Document excerpt:
{text}"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _cache_path(file_hash: str) -> Path:
    return CACHE_DIR / f"{file_hash}.json"

def _load_from_cache(file_hash: str):
    p = _cache_path(file_hash)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def _save_to_cache(file_hash: str, data) -> None:
    with open(_cache_path(file_hash), "w") as f:
        json.dump(data, f)

def _get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    loaders = {
        ".pdf":  PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt":  TextLoader,
        ".pptx": UnstructuredPowerPointLoader,
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")
    return loaders[ext](file_path)


# ── Auto-summary ──────────────────────────────────────────────────────────────

def _generate_summary(docs: List[Document]) -> str:
    """
    Generate a 2-3 sentence summary of the document using the first ~2000 chars.
    Falls back to a plain excerpt if Ollama is unavailable.
    """
    # Take first ~2000 chars from the document as a representative sample
    sample_text = " ".join(d.page_content for d in docs[:5])[:2000].strip()

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  MODEL_NAME,
                "prompt": SUMMARY_PROMPT.format(text=sample_text),
                "stream": False,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            summary = resp.json().get("response", "").strip()
            if summary:
                return summary
    except Exception:
        pass

    # Fallback: return first 300 chars of raw text
    return sample_text[:300] + "…"


# ── Core ingestion ────────────────────────────────────────────────────────────

def ingest_document(file_path: str):
    """
    Load a document and return (child_chunks, parent_chunks, from_cache, summary).
    """
    fhash    = _file_hash(file_path)
    cached   = _load_from_cache(fhash)
    filename = Path(file_path).name

    if cached:
        child_docs  = [Document(page_content=d["content"], metadata=d["metadata"]) for d in cached["children"]]
        parent_docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in cached["parents"]]
        summary     = cached.get("summary", "")
        return child_docs, parent_docs, True, summary

    loader   = _get_loader(file_path)
    raw_docs = loader.load()

    for i, doc in enumerate(raw_docs):
        doc.metadata.update({
            "source":    filename,
            "file_path": file_path,
            "file_hash": fhash,
            "page":      doc.metadata.get("page", i),
        })

    # ── Generate summary before chunking ─────────────────────────────────────
    summary = _generate_summary(raw_docs)

    # ── Parent chunks ─────────────────────────────────────────────────────────
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    parent_docs = parent_splitter.split_documents(raw_docs)
    for i, doc in enumerate(parent_docs):
        doc.metadata["chunk_id"]   = f"{fhash}_p{i}"
        doc.metadata["chunk_type"] = "parent"

    # ── Child chunks ──────────────────────────────────────────────────────────
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    child_docs = child_splitter.split_documents(raw_docs)
    for i, doc in enumerate(child_docs):
        doc.metadata["chunk_id"]   = f"{fhash}_c{i}"
        doc.metadata["chunk_type"] = "child"

    _save_to_cache(fhash, {
        "children": [{"content": d.page_content, "metadata": d.metadata} for d in child_docs],
        "parents":  [{"content": d.page_content, "metadata": d.metadata} for d in parent_docs],
        "summary":  summary,
    })

    return child_docs, parent_docs, False, summary


def ingest_multiple(file_paths: List[str]):
    """Ingest a list of files; returns combined child+parent docs and a status report."""
    all_children, all_parents = [], []
    report = {}

    for fp in file_paths:
        try:
            children, parents, from_cache, summary = ingest_document(fp)
            all_children.extend(children)
            all_parents.extend(parents)
            report[Path(fp).name] = {
                "status":  "cached" if from_cache else "ingested",
                "chunks":  len(children),
                "pages":   len(set(d.metadata.get("page", 0) for d in children)),
                "summary": summary,
            }
        except Exception as e:
            report[Path(fp).name] = {"status": "error", "error": str(e)}

    return all_children, all_parents, report