#  Personal Knowledge Copilot

A production-grade **Retrieval-Augmented Generation (RAG)** system that lets you chat conversationally with your own documents — fully **offline**, powered by local LLMs via **Ollama**.

Upload PDFs, Word docs, PowerPoints, or plain text files, and get grounded, cited answers with source transparency and quality evaluation — all running on your machine.

---

##  Features

###  Smart Document Ingestion
- **Multi-format support**: PDF, DOCX, PPTX, TXT
- **Parent-Child chunking**: Small child chunks for precision retrieval; large parent chunks for rich LLM context
- **SHA-256 file caching**: Re-uploaded files are served instantly from cache
- **Auto-summarisation**: Each ingested document gets a 2–3 sentence summary (via Mistral/Ollama)

###  Advanced Retrieval Pipeline
- **Hybrid retrieval (BM25 + FAISS)**: Combines dense semantic search with sparse keyword matching using Reciprocal Rank Fusion (RRF)
- **Multi-query expansion**: The user's question is automatically rephrased 3 ways to improve document coverage
- **HyDE query expansion** *(optional toggle)*: Generates a hypothetical answer to improve dense retrieval signal
- **CrossEncoder re-ranking**: Retrieved candidates are re-scored with `cross-encoder/ms-marco-MiniLM-L-6-v2` for precision
- **Parent-chunk retrieval**: LLM sees full context passages, not tiny snippets

###  Conversational Chat
- **Streaming token output** with a live typing cursor
- **Sliding-window conversation memory** (last 6 turns)
- **Inline source citations**: Every answer shows `[Source: filename, p.N]`
- **Answer grounding score**: Cosine similarity between the answer and the retrieved context (🟢 High / 🟡 Medium / 🔴 Low)
- **Graceful no-context fallback**: Refuses to hallucinate when context is too weak
- **Chat export**: Download the full conversation as a `.md` file

###  RAG Evaluation Dashboard
- **4 RAGAS metrics**: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **Auto-generates eval questions** from your indexed documents using Mistral — no manual dataset needed
- **Manual eval dataset support**: Provide your own `evaluation_dataset.json` for repeatable benchmarks
- **Per-question breakdown table** with downloadable JSON results
- **Hybrid evaluation**: Custom Ollama-based scoring for Faithfulness & Context Precision; RAGAS + Llama3 for Answer Relevancy & Context Recall

### Document Library Management
- View all indexed documents with chunk counts
- Remove individual documents from the index (with live re-index)
- Per-document summaries shown in the sidebar

---

##  Tech Stack

| Layer | Technology |
|---|---|
| **UI** | [Streamlit](https://streamlit.io/) |
| **LLM (local)** | [Ollama](https://ollama.ai/) — `mistral` (chat) + `llama3` (eval) |
| **Embeddings** | [Sentence Transformers](https://www.sbert.net/) — `BAAI/bge-small-en-v1.5` |
| **Dense index** | [FAISS](https://faiss.ai/) — `IndexFlatIP` (cosine via L2-normed vectors) |
| **Sparse index** | [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25Okapi |
| **Re-ranking** | [CrossEncoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) — `ms-marco-MiniLM-L-6-v2` |
| **Doc loaders** | [LangChain Community](https://python.langchain.com/) — PyPDF, Docx2txt, UnstructuredPPTX |
| **Evaluation** | [RAGAS](https://docs.ragas.io/) 0.4.x |

---

##  Project Structure

```
Personal-Knowledge-Copilot/
├── app.py               # Streamlit UI — sidebar, chat tab, evaluation tab
├── data_ingestion.py    # File loading, parent-child chunking, caching, summarisation
├── embeddings.py        # KnowledgeIndex: FAISS + BM25 hybrid retrieval, persistence
├── rag.py               # RAGPipeline: multi-query, HyDE, re-ranking, streaming, memory
├── evaluation.py        # RAGAS evaluation + custom Ollama-based metrics
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .cache/              # Auto-created: FAISS index, BM25, ingestion cache, eval results
```

---

##  Setup & Installation

###  Clone the repo
```bash
git clone https://github.com/SpandanNagale/Personal-Knowledge-Assistant
cd Personal-Knowledge-Assistant
```

###  Create & activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

###  Install Python dependencies
```bash
pip install -r requirements.txt
```

###  Install & start Ollama
Download Ollama from [ollama.ai](https://ollama.ai), then pull the required models:
```bash
ollama pull mistral    # Used for chat, summarisation & auto-eval question generation
ollama pull llama3     # Used by RAGAS for Answer Relevancy & Context Recall scoring
```

Ensure Ollama is running (it starts automatically on most systems, or run `ollama serve`).

### 5 Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

##  Evaluation

### Option A — Auto-generated (zero config)
Simply index a document and click ** Run Evaluation**. The app will:
1. Sample text chunks from your indexed documents
2. Ask Mistral to generate factual question + ground-truth answer pairs
3. Run the full RAG pipeline on each question
4. Score results with RAGAS + custom Ollama judges

### Option B — Manual dataset
Create an `evaluation_dataset.json` in the project root:
```json
[
  {
    "question": "What is the main topic of the document?",
    "ground_truth": "The document covers transformer architecture advances including BERT and GPT."
  }
]
```
The app will detect this file and use it instead of auto-generating questions.

### Metrics explained
| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? (0 = hallucinated, 1 = fully supported) |
| **Answer Relevancy** | Does the answer actually address the question asked? |
| **Context Precision** | What fraction of retrieved chunks are relevant to the question? |
| **Context Recall** | Did we retrieve all the information needed to answer correctly? |

Results are saved to `.cache/eval_results.json` and can be downloaded from the UI.

---

##  Configuration

Key constants you can tune in each module:

| File | Constant | Default | Description |
|---|---|---|---|
| `rag.py` | `MODEL_NAME` | `mistral` | Ollama model for chat |
| `rag.py` | `NUM_QUERIES` | `3` | Query variants for multi-query retrieval |
| `rag.py` | `RETRIEVE_TOP_K` | `10` | Candidates retrieved per query variant |
| `rag.py` | `RERANK_TOP_K` | `3` | Final chunks after CrossEncoder re-ranking |
| `rag.py` | `CONTEXT_WINDOW` | `6` | Conversation history turns kept in memory |
| `embeddings.py` | `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence embedding model |
| `data_ingestion.py` | `CHILD_CHUNK_SIZE` | `400` | Token size for retrieval chunks |
| `data_ingestion.py` | `PARENT_CHUNK_SIZE` | `1500` | Token size for LLM context chunks |
| `evaluation.py` | `RAGAS_EVAL_MODEL` | `llama3` | Ollama model for RAGAS scoring |

---

##  Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally with `mistral` and `llama3` models pulled
- ~2 GB disk space for models and FAISS indexes

---

##  Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.
