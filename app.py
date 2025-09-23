import streamlit as st
from data_ingestion import DataIngestion
from embeddings import Embeddings
from rag import RAG_ans

# ------------------------------
# Load & Cache Data/Embeddings
# ------------------------------
@st.cache_resource
def load_index(file_path="Advances in Transformer Architectures_ Integrating BERT, GPT, and Vision Models (1)_final.pdf"):
    data = DataIngestion(file_path).process_file()
    embed_model = Embeddings()
    embed_model.fit(data)
    return embed_model

st.set_page_config(page_title="Personal Knowledge Copilot", layout="wide")
st.title("🤖 Personal Knowledge Copilot")

# Sidebar for file upload
with st.sidebar:
    st.header("📂 Knowledge Source")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf","pptx","txt","docx"])
    if uploaded_file:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        embed_model = load_index("uploaded.pdf")
    else:
        st.info("Using default file: apjspeech.pdf")
        embed_model = load_index()

# ------------------------------
# Main UI
# ------------------------------
query = st.text_input("Ask me anything:", placeholder="Type your question here...")

if st.button("Get Answer") or query:
    if query.strip():
        with st.spinner("🔎 Searching..."):
            answer = RAG_ans(embed_model, question=query, k=3)
            results = embed_model.search(query, k=3)

        # Show answer
        st.subheader("💡 Answer")
        st.write(answer)

        # Show retrieved context
        with st.expander("📂 Retrieved Context"):
            for res in results:
                st.markdown(f"**Score:** {res['score']:.4f}")
                st.write(res["text"])
                st.caption(res["metadata"])
                st.markdown("---")
