import ollama
import json
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from src.config import MODEL_NAME, MAX_REFLECTION_CYCLES, GRADER_THRESHOLD, OLLAMA_BASE_URL
from src.retrieval.retriever import Retriever

class AgentState(TypedDict):
    query: str
    context: str
    answer: str
    grade: str
    score: float
    cycles: int
    feedback: str
    final_answer: str

def retrieve_node(state: AgentState, retriever: Retriever) -> AgentState:
    query = state["query"]
    context = retriever.get_context(query)
    return {"context": context}


def generate_node(state: AgentState) -> AgentState:
    query = state["query"]
    context = state["context"]
    cycles = state.get("cycles", 0) + 1
    feedback = state.get("feedback", "")

    
    prompt_parts = [
        "You are an AI assistant helping with academic questions.",
        "Answer based ONLY on the retrieved context below.",
        "If the context does not contain enough information, say so clearly.",
        "",
        "Retrieved Context:",
        context,
        "",
        "Question:",
        query,
    ]
    
    if feedback:
        prompt_parts.extend([
            "",
            "Previous attempt feedback:",
            feedback,
            "",
            "Please improve your answer based on the feedback above.",
        ])
    
    prompt_parts.append("Answer:")
    prompt = "\n".join(prompt_parts)
    
    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"].strip()
    
    return {"answer": answer, "cycles": cycles}


def grade_node(state: AgentState) -> AgentState:
    query = state["query"]
    context = state["context"]
    answer = state["answer"]
    cycles = state.get("cycles", 0)
    
    prompt = f"""You are a strict grader evaluating the quality of an AI-generated answer.

Question: {query}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate STRICTLY on these criteria:
1. HALLUCINATION: Does the answer make ANY claims NOT explicitly supported by the context? (yes/no)
2. SUPPORT: Is EVERY claim in the answer backed by the context? (yes/no)
3. RELEVANCE: Does the answer actually address ALL aspects of the question? (yes/no)
4. COMPLETENESS: Is the answer comprehensive enough or missing important details? (yes/no)

Score Guidelines:
- 0.9-1.0: Perfect answer, fully supported, no hallucinations
- 0.7-0.8: Good answer with minor issues or missing details
- 0.5-0.6: Acceptable but has some unsupported claims or lacks completeness
- 0.0-0.4: Poor answer with hallucinations or major gaps

Respond with ONLY a valid JSON object:
{{
  "hallucination": "yes" or "no",
  "supported": "yes" or "no",
  "relevant": "yes" or "no",
  "complete": "yes" or "no",
  "score": 0.0 to 1.0,
  "feedback": "brief explanation of issues and improvements needed"
}}

JSON:"""
    
    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    grade_text = response["message"]["content"].strip()
    
    try:
        if "```json" in grade_text:
            grade_text = grade_text.split("```json")[1].split("```")[0]
        elif "```" in grade_text:
            grade_text = grade_text.split("```")[1].split("```")[0]
        grade = json.loads(grade_text)
    except (json.JSONDecodeError, IndexError):
        grade = {
            "hallucination": "unknown",
            "supported": "unknown",
            "relevant": "unknown",
            "score": 0.5,
            "feedback": "Could not parse grade response."
        }
    
    return {
        "grade": grade_text,
        "score": grade.get("score", 0.5),
        "feedback": grade.get("feedback", "No feedback"),
        "cycles": cycles 
    }


def should_regenerate(state: AgentState) -> Literal["regenerate", "end"]:
    score = state.get("score", 0.0)
    cycles = state.get("cycles", 0)
    
    if score >= GRADER_THRESHOLD or cycles >= MAX_REFLECTION_CYCLES:
        return "end"
    return "regenerate"


def build_reflective_graph(retriever: Retriever) -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", lambda s: retrieve_node(s, retriever))
    graph.add_node("generate", generate_node)
    graph.add_node("grade", grade_node)
    
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade")
    graph.add_conditional_edges(
        "grade",
        should_regenerate,
        {
            "regenerate": "generate",
            "end": END
        }
    )
    
    return graph.compile()


class ReflectiveAgent:
    def __init__(self, retriever: Retriever = None):
        self.retriever = retriever or Retriever()
        self.graph = build_reflective_graph(self.retriever)
    
    def answer(self, query: str) -> dict:
        initial_state = {
            "query": query,
            "context": "",
            "answer": "",
            "grade": "",
            "score": 0.0,
            "cycles": 0,
            "feedback": "",
            "final_answer": ""
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "answer": result["answer"],
            "context": result["context"],
            "grade": result.get("grade", ""),
            "score": result.get("score", 0.0),
            "cycles": result.get("cycles", 0),
            "feedback": result.get("feedback", ""),
            "method": "reflective_agent"
        }


if __name__ == "__main__":
    agent = ReflectiveAgent()
    result = agent.answer("What is Retrieval Augmented Generation?")
    print(f"Answer: {result['answer']}")
    print(f"Score: {result['score']}")
    print(f"Cycles: {result['cycles']}")
