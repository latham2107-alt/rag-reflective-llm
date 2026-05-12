import sys
from src.retrieval.retriever import Retriever
from src.generation.linear_rag import LinearRAG
from src.generation.reflective_agent import ReflectiveAgent

def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def run_demo():
    print_separator("Linear RAG vs Reflective Agent - CLI Demo")
    print("\nThis demo compares two approaches:")
    print("  1. Linear RAG:   retrieve -> generate -> return")
    print("  2. Reflective:   retrieve -> generate -> grade -> regenerate (up to 3)")
    print("\nType your question. Type 'quit' to exit.")
    print("-" * 60)
    
    try:
        retriever = Retriever()
    except Exception as e:
        print(f"\n[ERROR] Could not initialize retriever: {e}")
        print("Make sure you have built the index first:")
        print("  python -m src.indexing.build_index")
        sys.exit(1)
    
    linear_rag = LinearRAG(retriever=retriever)
    reflective = ReflectiveAgent(retriever=retriever)
    
    while True:
        try:
            query = input("\nYour question: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            
            print("-" * 60)
            print("\n[Linear RAG]")
            linear_result = linear_rag.answer(query)
            print(f"Answer: {linear_result['answer']}")
            print(f"Cycles: {linear_result['cycles']}")
            
            print("-" * 60)
            print("\n[Reflective Agent]")
            reflective_result = reflective.answer(query)
            print(f"Answer: {reflective_result['answer']}")
            print(f"Score: {reflective_result['score']:.2f}")
            print(f"Cycles: {reflective_result['cycles']}")
            if reflective_result.get("feedback"):
                print(f"Feedback: {reflective_result['feedback']}")
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    run_demo()
