# 📖 Personal Knowledge Copilot  

A lightweight **Retrieval-Augmented Generation (RAG)** system built with **LangChain components, FAISS, Sentence Transformers, Ollama, and Streamlit**.  
Upload a knowledge source (PDF, DOCX, PPTX, TXT), embed it, and query it conversationally with an LLM.  

---

## 🚀 Features
- **Multi-format ingestion**: PDF, Word, PowerPoint, and text files  
- **Chunking & cleaning** for efficient embeddings  
- **Embeddings with Sentence Transformers** + **FAISS index**  
- **Ollama LLM integration** for offline, local inference  
- **Streamlit UI** for an interactive chat experience  
- **Context transparency**: retrieved passages + similarity scores are displayed  

---

## 🛠️ Tech Stack
- [LangChain](https://www.langchain.com/) → ingestion, splitting  
- [Sentence Transformers](https://www.sbert.net/) → embeddings  
- [FAISS](https://faiss.ai/) → vector search  
- [Ollama](https://ollama.ai/) → local LLMs (e.g., `Gemma3`)  
- [Streamlit](https://streamlit.io/) → user interface  

---

## 📂 Project Structure
Personal-Knowledge-Copilot/
│── data_ingestion.py # File ingestion & chunking
│── embeddings.py # FAISS + SBERT embeddings
│── rag.py # RAG pipeline with Ollama LLM
│── app.py # Streamlit UI
│── requirements.txt # Python dependencies
│── README.md # Project documentation
│── sample_data/ # Example PDFs or docs

### 1️⃣ Clone the repo

- git clone https://github.com/SpandanNagale/Personal-Knowledge-Assistant
- cd personal-knowledge-copilot

### 2️⃣ Create & activate a virtual environment
- python -m venv myenv
- source myenv/bin/activate   # Mac/Linux
- myenv\Scripts\activate      # Windows


### 3️⃣ Install dependencies
pip install -r requirements.txt

### 4️⃣ Install & run Ollama
ollama pull Gemma3

### 5️⃣ Run the Streamlit app
streamlit run app.py


