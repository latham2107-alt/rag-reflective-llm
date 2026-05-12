# rag-reflective-llm
We build two parallel systems for the same academic QA task:  1. Linear RAG Assistant.  2. Reflective Agent (LangGraph).  Reflective Agent incorporates the same retrieval step like Linear RAG Assistant but adds a self-grade node. This node checks if claims are supported by the retrieved text and if citations are consistent. 
