import faiss
from sentence_transformers import SentenceTransformer
from data_ingestion import DataIngestion



class Embeddings :
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model=SentenceTransformer(model_name)
        self.index=None
        self.embeddings=None
        self.mapping=None

    def fit(self , structured_data):
        text=[item["text"] for item in structured_data]
        self.embeddings=self.model.encode(text , convert_to_numpy=True)
        d=self.embeddings.shape[1]
        self.index=faiss.IndexFlatL2(d)
        self.index.add(self.embeddings)
        self.mapping = {i: structured_data[i] for i in range(len(structured_data))}
        

        

    def search(self, query:str , k:int=5  ):
        q_vec=self.model.encode([query] , convert_to_numpy=True)
        D,I=self.index.search(q_vec,k)
        results = []
        for idx, score in zip(I[0], D[0]):
            results.append({
                "text": self.mapping[idx]["text"],
                "metadata": self.mapping[idx]["metadata"],
                "score": float(score)
            })
        return results

