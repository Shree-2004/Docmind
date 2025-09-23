import ollama
from embeddings import Embeddings
from data_ingestion import DataIngestion


def get_llm(prompt: str, system: str = "You are a precise research assistant.") -> str:
    # ⚠️ Make sure this matches an installed Ollama model (check with `ollama list`)
    model = "Gemma3"  
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )
    return resp["message"]["content"].strip()


def call_llm(prompt: str) -> str:
    return get_llm(prompt)


PROMPT_TMPL = """You are an assistant. Use the following context to answer the question. 
If the answer is not in the context, say "I don’t know".

Context:
{context}

Question:
{question}
"""


def RAG_ans(embedding_model: Embeddings, question: str, k: int = 4):
   
    hits = embedding_model.search(question, k)

    context_parts = []
    for item in hits:
        meta_info = f"[source: {item['metadata']}, score: {item['score']:.2f}]"
        context_parts.append(f"{item['text']} {meta_info}")
    
    context = "\n\n".join(context_parts)
    prompt = PROMPT_TMPL.format(question=question, context=context)

    # 🔹 LLM call
    answer = call_llm(prompt=prompt)
    return answer



'''data = DataIngestion("apjspeech.pdf").process_file()
index = Embeddings()
index.fit(data)
results = RAG_ans(index, question="whose speech is this ")
print(results)'''