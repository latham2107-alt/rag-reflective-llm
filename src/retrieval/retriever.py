from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List

from src.config import INDEX_DIR, TOP_K


class Retriever:
    def __init__(self, persist_dir: str = INDEX_DIR, top_k: int = TOP_K):
        from langchain_huggingface import HuggingFaceEmbeddings
        from src.config import EMBEDDING_MODEL
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings
        )
        self.top_k = top_k
    
    def retrieve(self, query: str, k: int = None) -> List[Document]:
        k = k or self.top_k
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def get_context(self, query: str, k: int = None) -> str:
        docs = self.retrieve(query, k=k)
        contexts = []
        for i, doc in enumerate(docs, 1):
            contexts.append(f"[Source {i}] {doc.page_content}")
        return "\n\n".join(contexts)


if __name__ == "__main__":
    # Simple test
    retriever = Retriever()
    results = retriever.retrieve("What is RAG?")
    for doc in results:
        print(doc.page_content[:200])
