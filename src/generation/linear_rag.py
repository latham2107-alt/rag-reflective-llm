import ollama
from src.config import MODEL_NAME, OLLAMA_BASE_URL
from src.retrieval.retriever import Retriever


class LinearRAG:
    def __init__(self, retriever: Retriever = None):
        self.retriever = retriever or Retriever()
        self.model = MODEL_NAME
        self.base_url = OLLAMA_BASE_URL
    
    def _build_prompt(self, query: str, context: str) -> str:
        prompt = f"""You are an AI assistant helping with academic questions. Answer based ONLY on the retrieved context below. If the context does not contain enough information to answer, say so clearly.

Retrieved Context:
{context}

Question: {query}

Answer:"""
        return prompt
    
    def answer(self, query: str) -> dict:
        context = self.retriever.get_context(query)
        prompt = self._build_prompt(query, context)
        client = ollama.Client(host=self.base_url)
        response = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response["message"]["content"].strip()
        
        return {
            "answer": answer,
            "context": context,
            "model": self.model,
            "cycles": 1,
            "method": "linear_rag"
        }


if __name__ == "__main__":
    rag = LinearRAG()
    result = rag.answer("What is Retrieval Augmented Generation?")
    print(f"Answer: {result['answer']}")
    print(f"Cycles used: {result['cycles']}")
