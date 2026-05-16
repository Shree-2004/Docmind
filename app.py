"""
app.py – Personal Knowledge Copilot  (Streamlit UI)
"""

import tempfile
from pathlib import Path

import streamlit as st

from data_ingestion import ingest_multiple
from embeddings import get_index
from rag import RAGPipeline
from evaluation import run_evaluation, load_last_results, load_eval_dataset

st.set_page_config(
    page_title="Personal Knowledge Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .citation-pill {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 12px;
        color: #94a3b8;
        margin: 2px 2px;
    }
    .grounding-bar {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin: 4px 2px;
    }
    .grounding-high { background: #14532d; color: #86efac; }
    .grounding-med  { background: #713f12; color: #fde68a; }
    .grounding-low  { background: #7f1d1d; color: #fca5a5; }
    .summary-box {
        background: #0f172a;
        border-left: 3px solid #6366f1;
        border-radius: 4px;
        padding: 8px 12px;
        font-size: 12px;
        color: #94a3b8;
        margin: 4px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {"messages": [], "rag": None, "lib_files": [], "summaries": {}}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

index = get_index()

if st.session_state["rag"] is None:
    st.session_state["rag"] = RAGPipeline(index, use_hyde=False)

rag: RAGPipeline = st.session_state["rag"]

def _sync_library():
    stats = index.stats()
    st.session_state["lib_files"] = [
        {"name": src, "chunks": cnt}
        for src, cnt in stats["sources"].items()
    ]

_sync_library()


def _grounding_badge(score: float) -> str:
    pct = int(score * 100)
    if score >= 0.7:
        cls, label = "grounding-high", "✅ High Grounding"
    elif score >= 0.4:
        cls, label = "grounding-med",  "⚠️ Medium Grounding"
    else:
        cls, label = "grounding-low",  "❌ Low Grounding"
    return f'<span class="grounding-bar {cls}">{label} {pct}%</span>'


def _score_color(score: float) -> str:
    if score != score:   return "⚪"   # NaN check
    if score >= 0.7:     return "🟢"
    elif score >= 0.4:   return "🟡"
    else:                return "🔴"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Knowledge Copilot")
    st.caption("Chat with your documents · Runs locally")
    st.divider()

    st.subheader("📂 Add Documents")
    uploaded = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "pptx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Ingest Documents", type="primary", disabled=not uploaded):
        saved_paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for uf in uploaded:
                dest = Path(tmpdir) / uf.name
                dest.write_bytes(uf.read())
                saved_paths.append(str(dest))

            with st.spinner("Processing & summarising documents…"):
                children, parents, report = ingest_multiple(saved_paths)
                if children:
                    index.add(children, parents)
                    _sync_library()

        for fname, info in report.items():
            if info["status"] == "error":
                st.error(f"❌ {fname}: {info['error']}")
            else:
                if info.get("summary"):
                    st.session_state["summaries"][fname] = info["summary"]
                label = "💾 cached" if info["status"] == "cached" else "✅ ingested"
                st.success(f"{label}: {fname} ({info['chunks']} chunks)")

    st.divider()

    st.subheader("📚 Document Library")
    lib = st.session_state["lib_files"]
    if not lib:
        st.caption("No documents indexed yet.")
    else:
        for doc in lib:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{doc['name']}**  \n`{doc['chunks']} chunks`")
            if col2.button("🗑", key=f"del_{doc['name']}", help="Remove from index"):
                removed = index.remove_source(doc["name"])
                st.session_state["summaries"].pop(doc["name"], None)
                _sync_library()
                st.toast(f"Removed {removed} chunks for {doc['name']}")
                st.rerun()

            summary = st.session_state["summaries"].get(doc["name"])
            if summary:
                st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

    st.divider()

    st.subheader("⚙️ Settings")
    use_hyde = st.toggle("HyDE query expansion", value=False)
    rag.use_hyde = use_hyde
    top_k = st.slider("Final chunks (after re-ranking)", min_value=1, max_value=6, value=3)

    if st.button("🧹 Clear conversation"):
        st.session_state["messages"] = []
        rag.clear_memory()
        st.rerun()

    st.divider()

    if st.session_state["messages"]:
        def _export_chat() -> str:
            lines = ["# Knowledge Copilot – Chat Export\n"]
            for msg in st.session_state["messages"]:
                role = "**You**" if msg["role"] == "user" else "**Copilot**"
                lines.append(f"{role}\n{msg['content']}\n")
                if msg.get("citations"):
                    lines.append("*Sources: " + ", ".join(
                        f"{c['source']} p.{c['page']}" for c in msg["citations"]
                    ) + "*\n")
            return "\n".join(lines)

        st.download_button(
            "⬇️ Export chat (.md)",
            data=_export_chat(),
            file_name="knowledge_copilot_chat.md",
            mime="text/markdown",
        )


# ── Main area with tabs ───────────────────────────────────────────────────────
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("## 💬 Chat with your documents")

    if not index.is_ready:
        st.info("👈 Upload and ingest a document from the sidebar to get started.")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                cit_html = " ".join(
                    f'<span class="citation-pill">📄 {c["source"]} · p.{c["page"]}</span>'
                    for c in msg["citations"]
                )
                if msg.get("grounding") is not None:
                    cit_html += " " + _grounding_badge(msg["grounding"])
                st.markdown(cit_html, unsafe_allow_html=True)
                if msg.get("queries") and len(msg["queries"]) > 1:
                    with st.expander("🔍 Search queries used"):
                        for q in msg["queries"]:
                            st.markdown(f"- {q}")

    if question := st.chat_input(
        "Ask anything about your documents…",
        disabled=not index.is_ready,
    ):
        st.session_state["messages"].append({"role": "user", "content": question, "citations": []})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            answer_placeholder   = st.empty()
            citation_placeholder = st.empty()

            answer_so_far   = []
            final_citations = []
            final_grounding = None
            final_queries   = []
            no_context_flag = False

            for chunk in rag.ask_stream(question, top_k=top_k):
                if isinstance(chunk, str):
                    answer_so_far.append(chunk)
                    answer_placeholder.markdown("".join(answer_so_far) + "▌")
                elif isinstance(chunk, dict):
                    if chunk["type"] == "citations":
                        final_citations = chunk["sources"]
                        final_grounding = chunk.get("grounding")
                        final_queries   = chunk.get("queries", [])
                    elif chunk["type"] == "no_context":
                        no_context_flag = True

            final_answer = (
                "I couldn't find relevant information in your documents. Try rephrasing or uploading more relevant files."
                if no_context_flag else "".join(answer_so_far)
            )
            answer_placeholder.markdown(final_answer)

            if final_citations:
                cit_html = " ".join(
                    f'<span class="citation-pill">📄 {c["source"]} · p.{c["page"]}</span>'
                    for c in final_citations
                )
                if final_grounding is not None:
                    cit_html += " " + _grounding_badge(final_grounding)
                citation_placeholder.markdown(cit_html, unsafe_allow_html=True)
                if final_queries and len(final_queries) > 1:
                    with st.expander("🔍 Search queries used"):
                        for q in final_queries:
                            st.markdown(f"- {q}")

        st.session_state["messages"].append({
            "role":      "assistant",
            "content":   final_answer,
            "citations": final_citations,
            "grounding": final_grounding,
            "queries":   final_queries,
        })


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("## 📊 RAG Evaluation Dashboard")
    st.caption("Measures answer quality using RAGAS metrics against your evaluation dataset.")

    eval_samples = load_eval_dataset()

    if not eval_samples:
        st.info(
            "No `evaluation_dataset.json` found — questions will be **auto-generated** "
            "from your indexed documents when you click Run Evaluation!"
        )
    else:
        st.success(f"✅ Found **{len(eval_samples)}** manual evaluation samples")
        st.markdown("**Sample questions:**")
        for s in eval_samples[:3]:
            st.markdown(f"- {s['question']}")
        if len(eval_samples) > 3:
            st.caption(f"...and {len(eval_samples) - 3} more")

    # Button always visible
    run_btn = st.button(
        "▶️ Run Evaluation",
        type="primary",
        disabled=not index.is_ready,
        help="Requires documents to be indexed first",
    )

    if run_btn:
        n_label = str(len(eval_samples)) if eval_samples else "auto-generating"
        with st.spinner(f"Running RAGAS evaluation ({n_label} questions)… this takes a few minutes"):
            results = run_evaluation(
                rag,
                eval_samples if eval_samples else None,
                index=index,
            )
            st.session_state["eval_results"] = results

    # ── Show results ──────────────────────────────────────────────────────────
    results = st.session_state.get("eval_results") or load_last_results()

    if results:
        if results.get("error"):
            st.error(f"Evaluation error: {results['error']}")
        else:
            scores = results.get("scores", {})

            st.divider()
            st.markdown("### Aggregate Scores")

            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                (c1, "Faithfulness",      scores.get("faithfulness",      0), "Answer grounded in context?"),
                (c2, "Answer Relevancy",  scores.get("answer_relevancy",  0), "Does answer address the question?"),
                (c3, "Context Precision", scores.get("context_precision", 0), "Retrieved chunks relevant?"),
                (c4, "Context Recall",    scores.get("context_recall",    0), "All info retrieved?"),
            ]
            for col, name, score, tip in metrics:
                with col:
                    safe = score if score == score else 0.0
                    st.metric(
                        label=f"{_score_color(safe)} {name}",
                        value=f"{int(safe * 100)}%",
                        help=tip,
                    )

            st.divider()
            st.markdown("### Per-Question Breakdown")

            per_q = results.get("per_question", [])
            if per_q:
                import pandas as pd
                df = pd.DataFrame(per_q)
                df = df.rename(columns={
                    "question":          "Question",
                    "faithfulness":      "Faithfulness",
                    "answer_relevancy":  "Answer Relevancy",
                    "context_precision": "Context Precision",
                    "context_recall":    "Context Recall",
                })
                st.dataframe(
                    df[["Question", "Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]],
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "⬇️ Download results (.json)",
                    data=open(".cache/eval_results.json").read(),
                    file_name="eval_results.json",
                    mime="application/json",
                )