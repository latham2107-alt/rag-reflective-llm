import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from src.config import QA_FILE, RESULTS_DIR, EVAL_RESULTS_FILE
from src.retrieval.retriever import Retriever
from src.generation.linear_rag import LinearRAG
from src.generation.reflective_agent import ReflectiveAgent


def load_qa_dataset(qa_path: Path = QA_FILE) -> list[dict]:
    qa_pairs = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))
    return qa_pairs

def compute_exact_match(pred: str, gold: str) -> bool:
    def normalize(text):
        return " ".join(text.lower().strip().split())
    return normalize(pred) == normalize(gold)

def compute_f1(pred: str, gold: str) -> float:
    def tokenize(text):
        return set(text.lower().split())
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    intersection = pred_tokens & gold_tokens
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_method(method_name: str, method_fn, qa_pairs: list[dict], retriever: Retriever) -> dict:
    results = []
    em_scores = []
    f1_scores = []
    cycles_list = []
    
    for qa in tqdm(qa_pairs, desc=f"Evaluating {method_name}"):
        query = qa["question"]
        gold = qa["answer"]
        
        try:
            if method_name == "linear_rag":
                rag = LinearRAG(retriever=retriever)
                output = rag.answer(query)
            else:
                agent = ReflectiveAgent(retriever=retriever)
                output = agent.answer(query)
            
            pred = output.get("answer", "")
            cycles = output.get("cycles", 1)
            
            em = compute_exact_match(pred, gold)
            f1 = compute_f1(pred, gold)
            
            results.append({
                "question": query,
                "gold": gold,
                "pred": pred,
                "exact_match": em,
                "f1": f1,
                "cycles": cycles,
                "method": method_name
            })
            em_scores.append(em)
            f1_scores.append(f1)
            cycles_list.append(cycles)
            
        except Exception as e:
            results.append({
                "question": query,
                "gold": gold,
                "pred": f"ERROR: {str(e)}",
                "exact_match": False,
                "f1": 0.0,
                "cycles": 0,
                "method": method_name,
                "error": str(e)
            })
    
    return {
        "method": method_name,
        "results": results,
        "avg_em": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "avg_cycles": sum(cycles_list) / len(cycles_list) if cycles_list else 0.0,
        "num_questions": len(qa_pairs)
    }


def run_evaluation(qa_path: Path = QA_FILE) -> dict:
    print("Loading QA dataset...")
    qa_pairs = load_qa_dataset(qa_path)
    
    print(f"Loaded {len(qa_pairs)} QA pairs.")

    print("Initializing retriever...")
    retriever = Retriever()
    
    print("\n" + "="*50)
    print("EVALUATION: Linear RAG")
    print("="*50)
    linear_results = evaluate_method("linear_rag", None, qa_pairs, retriever)
    print(f"Exact Match: {linear_results['avg_em']:.2%}")
    print(f"F1 Score: {linear_results['avg_f1']:.4f}")
    print(f"Avg Cycles: {linear_results['avg_cycles']:.2f}")
    
    print("\n" + "="*50)
    print("EVALUATION: Reflective Agent")
    print("="*50)
    reflective_results = evaluate_method("reflective_agent", None, qa_pairs, retriever)
    print(f"Exact Match: {reflective_results['avg_em']:.2%}")
    print(f"F1 Score: {reflective_results['avg_f1']:.4f}")
    print(f"Avg Cycles: {reflective_results['avg_cycles']:.2f}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(qa_pairs),
        "linear_rag": linear_results,
        "reflective_agent": reflective_results,
        "comparison": {
            "em_improvement": reflective_results["avg_em"] - linear_results["avg_em"],
            "f1_improvement": reflective_results["avg_f1"] - linear_results["avg_f1"],
            "cycles_overhead": reflective_results["avg_cycles"] - linear_results["avg_cycles"]
        }
    }
    
    with open(EVAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {EVAL_RESULTS_FILE}")
    
    return final_results


if __name__ == "__main__":
    run_evaluation()
