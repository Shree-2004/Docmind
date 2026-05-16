"""
evaluation.py – RAG Evaluation using RAGAS metrics.

Auto-generates evaluation questions from indexed documents so you
never need to manually maintain evaluation_dataset.json.

Metrics
───────
• Faithfulness        – Is the answer grounded in the retrieved context?
• Answer Relevancy    – Does the answer address the question?
• Context Precision   – Are the retrieved chunks relevant to the question?
• Context Recall      – Did we retrieve all necessary information?
"""

from __future__ import annotations

import json
import requests
from pathlib import Path
from typing import List, Dict

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOllama          # chat model — required by RAGAS
from langchain_community.embeddings import OllamaEmbeddings

from datasets import Dataset
from rag import RAGPipeline

# ── Config ────────────────────────────────────────────────────────────────────
EVAL_DATASET_PATH = Path("evaluation_dataset.json")
RESULTS_PATH      = Path(".cache/eval_results.json")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

OLLAMA_URL       = "http://localhost:11434/api/generate"
MODEL_NAME       = "mistral"   # used for chat & auto-question generation
RAGAS_EVAL_MODEL = "llama3"    # used for RAGAS scoring — strict JSON output required
                                # run: ollama pull llama3  (if not already pulled)

AUTO_GEN_PROMPT = """\
Read the following document excerpt and generate {n} factual questions that can be answered from it.
For each question also write the correct answer based strictly on the text.

Output ONLY valid JSON in this exact format, no extra text:
[
  {{"question": "...", "ground_truth": "..."}},
  {{"question": "...", "ground_truth": "..."}}
]

Document excerpt:
{text}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> float:
    if val is None:
        return 0.0
    if isinstance(val, list):
        valid = [v for v in val if v is not None and v == v]
        return round(float(sum(valid) / len(valid)), 3) if valid else 0.0
    try:
        f = float(val)
        return 0.0 if f != f else round(f, 3)   # NaN → 0.0
    except Exception:
        return 0.0


# ── Custom metric helpers (direct Ollama calls) ───────────────────────────────
# RAGAS 0.4.x uses the `instructor` library for structured outputs, which
# requires OpenAI tool-calling format — not supported by Ollama for all metrics.
# Faithfulness and Context Precision are computed here via simple prompts.

import re as _re

def _ollama_score(prompt: str) -> float:
    """Ask Ollama for a 0–1 score; return 0.0 on any failure."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=45,
        )
        if resp.status_code == 200:
            raw = resp.json().get("response", "").strip()
            # Accept "0.8", ".8", "1", "0" etc.
            match = _re.search(r'\b(1(\.0*)?|0?(\.[0-9]+))\b', raw)
            if match:
                return min(1.0, max(0.0, float(match.group())))
    except Exception:
        pass
    return 0.0


def _faithfulness(answer: str, contexts: List[str]) -> float:
    """Is the answer grounded in the retrieved context?"""
    ctx = "\n\n".join(contexts)[:3000]
    prompt = (
        "You are an evaluation judge. Score how faithful the ANSWER is to the CONTEXT.\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Reply with a SINGLE decimal number between 0.0 and 1.0.\n"
        "1.0 = answer fully supported by context\n"
        "0.5 = answer partially supported\n"
        "0.0 = answer not supported / contradicts context\n"
        "Score (number only):"
    )
    return _ollama_score(prompt)


def _context_precision(question: str, contexts: List[str]) -> float:
    """What fraction of retrieved chunks are actually relevant to the question?"""
    if not contexts:
        return 0.0
    scores = []
    for ctx in contexts:
        prompt = (
            "You are an evaluation judge. Score how RELEVANT the CONTEXT is for answering the QUESTION.\n\n"
            f"QUESTION: {question}\n\n"
            f"CONTEXT:\n{ctx[:1500]}\n\n"
            "Reply with a SINGLE decimal number between 0.0 and 1.0.\n"
            "1.0 = directly answers the question\n"
            "0.5 = somewhat related\n"
            "0.0 = unrelated\n"
            "Score (number only):"
        )
        scores.append(_ollama_score(prompt))
    return round(sum(scores) / len(scores), 3) if scores else 0.0

# ── Auto-generate eval questions ──────────────────────────────────────────────

def generate_eval_questions(index, n_per_doc: int = 3) -> List[Dict]:
    """
    Auto-generate question+ground_truth pairs from indexed documents.
    Samples a few chunks per document and asks Mistral to generate questions.
    """
    if not index.is_ready:
        return []

    all_samples = []
    stats       = index.stats()

    for source in stats["sources"]:
        # Get chunks for this source
        chunks = [
            d for d in index.child_docs
            if d["metadata"].get("source") == source
        ]

        if not chunks:
            continue

        # Sample up to 3 chunks from different pages for variety
        seen_pages = set()
        sampled    = []
        for chunk in chunks:
            page = chunk["metadata"].get("page", 0)
            if page not in seen_pages:
                seen_pages.add(page)
                sampled.append(chunk["content"])
            if len(sampled) >= 3:
                break

        sample_text = "\n\n".join(sampled)[:3000]

        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model":  MODEL_NAME,
                    "prompt": AUTO_GEN_PROMPT.format(n=n_per_doc, text=sample_text),
                    "stream": False,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                raw = resp.json().get("response", "").strip()
                # Strip markdown code fences if present
                raw = raw.replace("```json", "").replace("```", "").strip()
                samples = json.loads(raw)
                # Tag each sample with its source
                for s in samples:
                    s["source"] = source
                all_samples.extend(samples)
        except Exception:
            continue

    return all_samples


# ── Load eval dataset ─────────────────────────────────────────────────────────

def load_eval_dataset() -> List[Dict]:
    """Load manual evaluation_dataset.json if it exists."""
    if not EVAL_DATASET_PATH.exists():
        return []
    with open(EVAL_DATASET_PATH) as f:
        return json.load(f)


# ── Run evaluation ────────────────────────────────────────────────────────────

def run_evaluation(
    rag: RAGPipeline,
    samples: List[Dict] | None = None,
    index=None,
    auto_generate: bool = True,
) -> Dict:
    """
    Run RAGAS evaluation.
    If auto_generate=True and no manual samples, auto-generates from indexed docs.
    """
    # Prefer manual samples if provided
    if samples is None:
        samples = load_eval_dataset()

    # Fall back to auto-generated if no manual dataset
    if not samples and auto_generate and index is not None:
        samples = generate_eval_questions(index)

    if not samples:
        return {"error": "No evaluation samples available. Index a document first."}

    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []

    for sample in samples:
        question     = sample.get("question", "")
        ground_truth = sample.get("ground_truth", "")

        answer, citations = rag.ask(question)
        context_list = [c.get("excerpt", "") for c in citations if c.get("excerpt")]
        if not context_list:
            context_list = ["No context retrieved"]

        questions.append(question)
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(ground_truth)

    eval_dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    # ── Custom: Faithfulness & Context Precision (bypass RAGAS 0.4.x instructor) ──
    faith_scores   = [_faithfulness(a, c)         for a, c in zip(answers, contexts)]
    cp_scores      = [_context_precision(q, c)    for q, c in zip(questions, contexts)]

    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_recall

        # RAGAS 0.4.x: use llama3 (strict JSON output) for the two RAGAS metrics.
        ragas_llm = LangchainLLMWrapper(ChatOllama(model=RAGAS_EVAL_MODEL, temperature=0))
        ragas_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=RAGAS_EVAL_MODEL))

        answer_relevancy.llm        = ragas_llm
        answer_relevancy.embeddings = ragas_emb
        context_recall.llm          = ragas_llm

        result = evaluate(
            eval_dataset,
            metrics=[answer_relevancy, context_recall],
        )

        df = result.to_pandas()

        ar_scores = list(df["answer_relevancy"])
        cr_scores = list(df["context_recall"])

        scores = {
            "faithfulness":      _safe_float(faith_scores),
            "answer_relevancy":  _safe_float(ar_scores),
            "context_precision": _safe_float(cp_scores),
            "context_recall":    _safe_float(cr_scores),
        }

        per_question = []
        for i in range(len(questions)):
            per_question.append({
                "question":          questions[i],
                "answer":            answers[i],
                "source":            samples[i].get("source", ""),
                "faithfulness":      round(faith_scores[i], 3),
                "answer_relevancy":  _safe_float(ar_scores[i] if i < len(ar_scores) else None),
                "context_precision": round(cp_scores[i], 3),
                "context_recall":    _safe_float(cr_scores[i] if i < len(cr_scores) else None),
            })

        output = {
            "scores":         scores,
            "per_question":   per_question,
            "auto_generated": not bool(load_eval_dataset()),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()   # prints full stack trace to Streamlit terminal
        # RAGAS failed — still return the custom-computed metrics
        scores = {
            "faithfulness":      _safe_float(faith_scores),
            "answer_relevancy":  0.0,
            "context_precision": _safe_float(cp_scores),
            "context_recall":    0.0,
        }
        per_question = [
            {
                "question":          questions[i],
                "answer":            answers[i],
                "source":            samples[i].get("source", ""),
                "faithfulness":      round(faith_scores[i], 3),
                "answer_relevancy":  0.0,
                "context_precision": round(cp_scores[i], 3),
                "context_recall":    0.0,
            }
            for i in range(len(questions))
        ]
        output = {
            "scores":         scores,
            "per_question":   per_question,
            "auto_generated": not bool(load_eval_dataset()),
            "ragas_error":    str(e),
        }


    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    return output


def load_last_results() -> Dict | None:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None