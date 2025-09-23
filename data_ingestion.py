from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader , UnstructuredWordDocumentLoader , UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataIngestion:
    def __init__(self , file_path:str):
        self.file_path=file_path
        self.documents=[]
        self.file_type=self.detect_filetype()
    
    def detect_filetype(self):
        ext = self.file_path.lower()
        if ext.endswith(".pdf"):
            return "pdf"
        elif ext.endswith(".docx"):
            return "docx"
        elif ext.endswith(".pptx"):
            return "ppt"
        elif ext.endswith(".txt"):
            return "txt"
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        

    def extract_text(self):
        if self.file_type=="pdf":
            loader=PyPDFLoader(self.file_path)
            self.documents.extend(loader.load())
        elif self.file_type=="docx":
            loader=UnstructuredWordDocumentLoader(self.file_path)
            self.documents.extend(loader.load())
        elif self.file_type=="PPT":
            loader=UnstructuredPowerPointLoader(self.file_path)
            self.documents.extend(loader.load())
        elif self.file_type=="text":
            with open(self.file_path, "r" , encoding="utf-8") as f :
                text=f.read()
                self.documents.append(Document(page_content=text,  metadata={"source": self.file_path}))
        return self.documents 
    

    def cleaning(self, text:str)->str:
        text=text.replace("\n"," ").strip()
        text=" ".join(text.split())
        
        return text
    
    def chunking(self , chunk_size=300 , chunk_overlap=50):
        splitter=RecursiveCharacterTextSplitter( chunk_size=chunk_size , chunk_overlap=chunk_overlap )
        return splitter.split_documents(self.documents)

    def process_file(self,chunk_size=300 , chunk_overlap=50):

        self.extract_text()

        chunks=self.chunking(chunk_size=chunk_size,chunk_overlap=chunk_overlap)

        structured=[]
        for i , doc in enumerate(chunks):
            structured.append({
                "chunk_id": i + 1,
                "text": self.cleaning(doc.page_content),
                "metadata": doc.metadata
            })
        return structured    

