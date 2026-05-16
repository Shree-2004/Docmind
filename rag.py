"""
rag.py – Core RAG pipeline.

Features
────────
• Multi-query retrieval    (question rephrased 3 ways → broader coverage)
• CrossEncoder re-ranking  (retrieved chunks re-scored for precision)
• Conversation memory      (sliding-window buffer)
• HyDE query expansion     (optional)
• Source citation          (returned as structured metadata)
• Answer grounding score   (cosine similarity between answer and context)
• Confidence threshold     (refuses to answer if context is too weak)
• Streaming generator      (yields tokens one-by-one for Streamlit)
"""

from __future__ import annotations

import json
from collections import deque
from typing import Generator, List, Tuple

import numpy as np
import requests
from sentence_transformers import CrossEncoder

from embeddings import KnowledgeIndex, embed_texts

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL       = "http://localhost:11434/api/generate"
MODEL_NAME       = "mistral"
CONTEXT_WINDOW   = 6
MIN_SCORE        = 0.004
RETRIEVE_TOP_K   = 10      # candidates per query variant
RERANK_TOP_K     = 3       # final chunks after CrossEncoder re-ranking
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NUM_QUERIES      = 3       # number of query variants to generate


# ── Singleton CrossEncoder ────────────────────────────────────────────────────
_reranker: CrossEncoder | None = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a knowledgeable assistant that answers questions ONLY from the provided document excerpts.

Rules:
1. Base your answer strictly on the CONTEXT below.
2. If the context does not contain enough information, say so clearly — do NOT invent facts.
3. Cite your sources inline using [Source: <filename>, p.<page>] notation.
4. Be concise and direct. Use markdown formatting where helpful.
5. When a previous conversation is given, use it to understand follow-up questions."""

HYDE_PROMPT = """\
Write a short paragraph that would be a plausible answer to the following question.
Do not add disclaimers. Be factual and direct.
Question: {question}"""

MULTI_QUERY_PROMPT = """\
Generate {n} different ways to ask the following question to improve document retrieval.
Each variation should use different wording but ask the same thing.
Output ONLY the questions, one per line, no numbering, no extra text.

Original question: {question}"""


def _format_context(chunks: List[Tuple]) -> str:
    parts = []
    for doc, score in chunks:
        meta    = doc.metadata
        source  = meta.get("source", "unknown")
        page    = meta.get("page", "?")
        excerpt = doc.page_content.strip()
        parts.append(f"[Source: {source}, p.{page}]\n{excerpt}")
    return "\n\n---\n\n".join(parts)


def _format_history(history: deque) -> str:
    if not history:
        return ""
    lines = ["### Conversation History"]
    for q, a in history:
        lines.append(f"User: {q}\nAssistant: {a}")
    return "\n\n".join(lines)


# ── HyDE ──────────────────────────────────────────────────────────────────────

def _hyde_expand(question: str) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": HYDE_PROMPT.format(question=question), "stream": False},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("response", question)
    except Exception:
        pass
    return question


# ── Multi-query generation ────────────────────────────────────────────────────

def _generate_query_variants(question: str, n: int = NUM_QUERIES) -> List[str]:
    """
    Ask the LLM to rephrase the question N different ways.
    Returns a list of query strings including the original.
    Falls back to just the original if LLM call fails.
    """
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  MODEL_NAME,
                "prompt": MULTI_QUERY_PROMPT.format(n=n, question=question),
                "stream": False,
            },
            timeout=20,
        )
        if resp.status_code == 200:
            raw       = resp.json().get("response", "")
            variants  = [line.strip() for line in raw.strip().splitlines() if line.strip()]
            variants  = variants[:n]   # cap at n
            # Always include the original question
            all_queries = [question] + [v for v in variants if v != question]
            return all_queries[:n + 1]
    except Exception:
        pass
    return [question]   # fallback


# ── Multi-query retrieval ─────────────────────────────────────────────────────

def _multi_query_retrieve(
    question: str,
    index: KnowledgeIndex,
    top_k: int = RETRIEVE_TOP_K,
) -> List[Tuple]:
    """
    Generate multiple query variants, retrieve for each, then deduplicate
    and merge results using chunk_id as the unique key.
    """
    queries   = _generate_query_variants(question)
    seen_ids  = set()
    all_hits  = []

    for q in queries:
        hits = index.query(q, top_k=top_k, use_parent=True)
        for doc, score in hits:
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_hits.append((doc, score))

    # Sort by score descending before passing to re-ranker
    all_hits.sort(key=lambda x: x[1], reverse=True)
    return all_hits


# ── CrossEncoder re-ranking ───────────────────────────────────────────────────

def _rerank(question: str, hits: List[Tuple], top_k: int = RERANK_TOP_K) -> List[Tuple]:
    """Re-score with CrossEncoder and return top_k best chunks."""
    if not hits:
        return hits

    reranker = _get_reranker()
    pairs    = [(question, doc.page_content) for doc, _ in hits]
    scores   = reranker.predict(pairs)

    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    return [hit for hit, _ in ranked[:top_k]]


# ── Answer grounding score ────────────────────────────────────────────────────

def _grounding_score(answer: str, chunks: List[Tuple]) -> float:
    """Cosine similarity between answer embedding and context embedding."""
    try:
        context_text = " ".join(doc.page_content for doc, _ in chunks)
        vecs         = embed_texts([answer, context_text])
        a, b         = vecs[0], vecs[1]
        score        = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        return round(max(0.0, min(1.0, score)), 3)
    except Exception:
        return 0.0


# ── Streaming call to Ollama ──────────────────────────────────────────────────

def _stream_ollama(prompt: str) -> Generator[str, None, None]:
    try:
        with requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": True},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️ **Cannot connect to Ollama.** Make sure it is running (`ollama serve`)."
    except Exception as e:
        yield f"\n\n⚠️ **Error:** {e}"


# ── Main RAG class ────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, index: KnowledgeIndex, use_hyde: bool = False):
        self.index    = index
        self.use_hyde = use_hyde
        self.memory: deque[Tuple[str, str]] = deque(maxlen=CONTEXT_WINDOW)

    def clear_memory(self) -> None:
        self.memory.clear()

    def ask_stream(
        self,
        question: str,
        top_k: int = RERANK_TOP_K,
    ) -> Generator[str | dict, None, None]:
        """
        Streaming generator that yields:
          - str tokens
          - {"type": "citations", "sources": [...], "grounding": float, "queries": [...]}
          - {"type": "no_context"} if confidence is too low
        """
        if not self.index.is_ready:
            yield "⚠️ No documents indexed yet. Please upload a document first."
            return

        # ── Step 1: HyDE (optional) ───────────────────────────────────────────
        base_query = _hyde_expand(question) if self.use_hyde else question

        # ── Step 2: Multi-query retrieval ─────────────────────────────────────
        hits    = _multi_query_retrieve(base_query, self.index, top_k=RETRIEVE_TOP_K)
        queries = _generate_query_variants(question)   # for display

        if not hits or hits[0][1] < MIN_SCORE:
            yield {"type": "no_context"}
            return

        # ── Step 3: CrossEncoder re-ranking ───────────────────────────────────
        reranked_hits = _rerank(question, hits, top_k=top_k)

        context = _format_context(reranked_hits)
        history = _format_history(self.memory)

        # ── Step 4: Build prompt ──────────────────────────────────────────────
        prompt_parts = [SYSTEM_PROMPT, ""]
        if history:
            prompt_parts += [history, ""]
        prompt_parts += [
            "### Context",
            context,
            "",
            f"### Question\n{question}",
            "",
            "### Answer",
        ]
        full_prompt = "\n".join(prompt_parts)

        # ── Step 5: Stream answer ─────────────────────────────────────────────
        full_answer = []
        for token in _stream_ollama(full_prompt):
            full_answer.append(token)
            yield token

        answer_text = "".join(full_answer)

        # ── Step 6: Update memory ─────────────────────────────────────────────
        self.memory.append((question, answer_text))

        # ── Step 7: Grounding score ───────────────────────────────────────────
        grounding = _grounding_score(answer_text, reranked_hits)

        # ── Step 8: Emit metadata ─────────────────────────────────────────────
        sources = []
        seen    = set()
        for doc, score in reranked_hits:
            meta = doc.metadata
            key  = (meta.get("source"), meta.get("page"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source":  meta.get("source", "unknown"),
                    "page":    meta.get("page", "?"),
                    "score":   round(score, 4),
                    "excerpt": doc.page_content[:200].strip(),
                })

        yield {
            "type":      "citations",
            "sources":   sources,
            "grounding": grounding,
            "queries":   queries,   # ← the variants used for retrieval
        }

    def ask(self, question: str, top_k: int = RERANK_TOP_K) -> Tuple[str, List[dict]]:
        """Non-streaming version. Returns (answer_text, citations_list)."""
        answer_tokens: List[str] = []
        citations: List[dict]    = []
        no_context               = False

        for chunk in self.ask_stream(question, top_k=top_k):
            if isinstance(chunk, str):
                answer_tokens.append(chunk)
            elif isinstance(chunk, dict):
                if chunk.get("type") == "citations":
                    citations = chunk["sources"]
                elif chunk.get("type") == "no_context":
                    no_context = True

        if no_context:
            return (
                "I couldn't find relevant information in your documents to answer this question.",
                [],
            )

        return "".join(answer_tokens), citations