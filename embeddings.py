"""
embeddings.py – Hybrid retrieval (BM25 + FAISS) with persistent index.

Architecture
────────────
• Child chunks are embedded with BAAI/bge-small-en-v1.5 and stored in FAISS.
• BM25 index is built over the same child chunks for keyword matching.
• At query time both indexes are queried; scores are combined (RRF fusion).
• The top-k child chunk IDs are mapped back to their parent chunks so the
  LLM receives rich, full-context passages instead of tiny snippets.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────
INDEX_DIR   = Path(".cache/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_PATH  = INDEX_DIR / "faiss.index"
META_PATH   = INDEX_DIR / "metadata.json"
BM25_PATH   = INDEX_DIR / "bm25.pkl"
PARENT_PATH = INDEX_DIR / "parents.json"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # better retrieval than all-MiniLM
TOP_K       = 5                            # candidates per retriever
RRF_K       = 60                           # RRF constant


# ── Singleton model ───────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of strings; returns float32 numpy array."""
    model = _get_model()
    # BGE models benefit from a query/passage prefix
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)


# ── Index management ──────────────────────────────────────────────────────────

class KnowledgeIndex:
    """
    Wraps FAISS + BM25 for hybrid retrieval and maps hits to parent chunks.
    """

    def __init__(self):
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.bm25: BM25Okapi | None = None
        self.child_docs:  List[dict] = []   # metadata + content for each child
        self.parent_docs: List[dict] = []   # metadata + content for each parent

    # ── Build ────────────────────────────────────────────────────────────────

    def build(
        self,
        child_chunks: List[Document],
        parent_chunks: List[Document],
    ) -> None:
        """Index a fresh set of child + parent documents."""
        self.child_docs  = [{"content": d.page_content, "metadata": d.metadata}
                            for d in child_chunks]
        self.parent_docs = [{"content": d.page_content, "metadata": d.metadata}
                            for d in parent_chunks]

        # ── FAISS (dense) ────────────────────────────────────────────────────
        texts = [d["content"] for d in self.child_docs]
        vecs  = embed_texts(texts)
        dim   = vecs.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)   # inner-product on L2-normed = cosine
        self.faiss_index.add(vecs)

        # ── BM25 (sparse) ────────────────────────────────────────────────────
        tokenised = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenised)

        self.save()

    def add(
        self,
        child_chunks: List[Document],
        parent_chunks: List[Document],
    ) -> None:
        """Incrementally add documents to an existing index."""
        if self.faiss_index is None:
            self.build(child_chunks, parent_chunks)
            return

        new_children = [{"content": d.page_content, "metadata": d.metadata}
                        for d in child_chunks]
        new_parents  = [{"content": d.page_content, "metadata": d.metadata}
                        for d in parent_chunks]

        # Filter out already-indexed chunks (by chunk_id)
        existing_ids = {d["metadata"].get("chunk_id") for d in self.child_docs}
        new_children = [d for d in new_children
                        if d["metadata"].get("chunk_id") not in existing_ids]

        if not new_children:
            return   # nothing new

        self.child_docs.extend(new_children)
        self.parent_docs.extend(new_parents)

        # Rebuild FAISS + BM25 from scratch (fast enough at this scale)
        texts = [d["content"] for d in self.child_docs]
        vecs  = embed_texts(texts)
        dim   = vecs.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(vecs)
        tokenised = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenised)

        self.save()

    # ── Query ────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = TOP_K,
        use_parent: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid retrieval: BM25 + FAISS combined with Reciprocal Rank Fusion.
        Returns list of (Document, score) sorted by relevance, highest first.
        When use_parent=True the returned Document contains the richer parent chunk.
        """
        if self.faiss_index is None or not self.child_docs:
            return []

        n = len(self.child_docs)
        k = min(top_k, n)

        # ── Dense retrieval ──────────────────────────────────────────────────
        q_vec = embed_texts([f"Represent this sentence for searching relevant passages: {question}"])
        _, faiss_idxs = self.faiss_index.search(q_vec, k)
        faiss_ranks   = {int(idx): rank for rank, idx in enumerate(faiss_idxs[0])}

        # ── Sparse retrieval ─────────────────────────────────────────────────
        bm25_scores = self.bm25.get_scores(question.lower().split())
        bm25_idxs   = np.argsort(bm25_scores)[::-1][:k]
        bm25_ranks  = {int(idx): rank for rank, idx in enumerate(bm25_idxs)}

        # ── RRF fusion ───────────────────────────────────────────────────────
        all_idxs = set(faiss_ranks) | set(bm25_ranks)
        rrf: dict[int, float] = {}
        for idx in all_idxs:
            dense_score  = 1.0 / (RRF_K + faiss_ranks.get(idx, k + 1))
            sparse_score = 1.0 / (RRF_K + bm25_ranks.get(idx, k + 1))
            rrf[idx] = dense_score + sparse_score

        top_idxs = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:k]

        # ── Map child → parent ───────────────────────────────────────────────
        results: List[Tuple[Document, float]] = []
        seen_sources: set[str] = set()

        for idx in top_idxs:
            child_meta = self.child_docs[idx]["metadata"]
            score      = rrf[idx]

            if use_parent:
                doc = self._get_parent_for_child(child_meta) or Document(
                    page_content=self.child_docs[idx]["content"],
                    metadata=child_meta,
                )
            else:
                doc = Document(
                    page_content=self.child_docs[idx]["content"],
                    metadata=child_meta,
                )

            # Deduplicate by (source, page)
            key = f"{child_meta.get('source')}_{child_meta.get('page')}"
            if key not in seen_sources:
                seen_sources.add(key)
                results.append((doc, score))

        return results

    def _get_parent_for_child(self, child_meta: dict) -> Document | None:
        """Find the best-matching parent chunk for a given child's metadata."""
        source = child_meta.get("source")
        page   = child_meta.get("page")
        for p in self.parent_docs:
            if p["metadata"].get("source") == source and p["metadata"].get("page") == page:
                return Document(page_content=p["content"], metadata=p["metadata"])
        # Fallback: same source, any page
        for p in self.parent_docs:
            if p["metadata"].get("source") == source:
                return Document(page_content=p["content"], metadata=p["metadata"])
        return None

    def remove_source(self, filename: str) -> int:
        """Remove all chunks belonging to a file; returns number of chunks removed."""
        before = len(self.child_docs)
        self.child_docs  = [d for d in self.child_docs
                            if d["metadata"].get("source") != filename]
        self.parent_docs = [d for d in self.parent_docs
                            if d["metadata"].get("source") != filename]
        removed = before - len(self.child_docs)

        if removed > 0 and self.child_docs:
            # Rebuild indexes
            texts = [d["content"] for d in self.child_docs]
            vecs  = embed_texts(texts)
            dim   = vecs.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(vecs)
            tokenised = [t.lower().split() for t in texts]
            self.bm25 = BM25Okapi(tokenised)
        elif not self.child_docs:
            self.faiss_index = None
            self.bm25 = None

        self.save()
        return removed

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(FAISS_PATH))
        with open(META_PATH,   "w") as f:
            json.dump(self.child_docs, f)
        with open(PARENT_PATH, "w") as f:
            json.dump(self.parent_docs, f)
        with open(BM25_PATH, "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self) -> bool:
        """Load from disk. Returns True if successful."""
        try:
            if not all(p.exists() for p in [FAISS_PATH, META_PATH, BM25_PATH, PARENT_PATH]):
                return False
            self.faiss_index = faiss.read_index(str(FAISS_PATH))
            with open(META_PATH)   as f: self.child_docs  = json.load(f)
            with open(PARENT_PATH) as f: self.parent_docs = json.load(f)
            with open(BM25_PATH, "rb") as f: self.bm25 = pickle.load(f)
            return True
        except Exception:
            return False

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        sources = {}
        for d in self.child_docs:
            src = d["metadata"].get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return {
            "total_child_chunks":  len(self.child_docs),
            "total_parent_chunks": len(self.parent_docs),
            "sources": sources,
        }

    @property
    def is_ready(self) -> bool:
        return self.faiss_index is not None and len(self.child_docs) > 0


# ── Module-level singleton ─────────────────────────────────────────────────────

_index: KnowledgeIndex | None = None

def get_index() -> KnowledgeIndex:
    global _index
    if _index is None:
        _index = KnowledgeIndex()
        _index.load()
    return _index