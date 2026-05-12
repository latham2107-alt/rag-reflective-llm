import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.retrieval.retriever import Retriever
from src.generation.linear_rag import LinearRAG
from src.generation.reflective_agent import ReflectiveAgent

class QueryRequest(BaseModel):
    question: str
    method: Optional[str] = "both" 

class QueryResponse(BaseModel):
    question: str
    linear_rag: Optional[Dict] = None
    reflective_agent: Optional[Dict] = None
    comparison: Optional[Dict] = None
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    retriever_ready: bool
    models_ready: bool
    startup_time: str
    uptime_seconds: float


retriever = None
linear_rag = None
reflective_agent = None
server_start_time = time.time()
server_start_iso = datetime.now().isoformat()

app = FastAPI(
    title="RAG vs Reflective Agent API",
    description="Compare Linear RAG and Reflective Agent approaches for academic QA",
    version="1.0.0"
)

templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(templates_dir))

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def initialize_components():
    global retriever, linear_rag, reflective_agent

    try:
        if retriever is None:
            retriever = Retriever()
        if linear_rag is None:
            linear_rag = LinearRAG(retriever=retriever)
        if reflective_agent is None:
            reflective_agent = ReflectiveAgent(retriever=retriever)
        return True
    except Exception as e:
        print(f"Failed to initialize components: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    initialize_components()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    retriever_ready = retriever is not None
    models_ready = linear_rag is not None and reflective_agent is not None

    status = "healthy" if (retriever_ready and models_ready) else "unhealthy"
    uptime_seconds = time.time() - server_start_time

    return HealthResponse(
        status=status,
        retriever_ready=retriever_ready,
        models_ready=models_ready,
        startup_time=server_start_iso,
        uptime_seconds=round(uptime_seconds, 2)
    )



@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not initialize_components():
        raise HTTPException(status_code=503, detail="Components not ready")

    start_time = time.time()
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = {
        "question": question,
        "linear_rag": None,
        "reflective_agent": None,
        "comparison": None,
        "processing_time": 0.0
    }

    try:
        if request.method in ["linear", "both"]:
            linear_result = await asyncio.get_event_loop().run_in_executor(
                None, linear_rag.answer, question
            )
            retrieved_docs = []
            if "context" in linear_result:
                context_parts = linear_result["context"].split("\n\n")
                for part in context_parts:
                    if part.strip() and part.startswith("[Source"):
                        retrieved_docs.append(part.strip())
            
            result["linear_rag"] = {
                "answer": linear_result.get("answer", ""),
                "cycles": linear_result.get("cycles", 1),
                "retrieved_docs": retrieved_docs,
                "confidence": linear_result.get("score", 0.8), 
                "grade": linear_result.get("grade", "N/A")
            }

        if request.method in ["reflective", "both"]:
            reflective_result = await asyncio.get_event_loop().run_in_executor(
                None, reflective_agent.answer, question
            )
            retrieved_docs = []
            if "context" in reflective_result:
                context_parts = reflective_result["context"].split("\n\n")
                for part in context_parts:
                    if part.strip() and part.startswith("[Source"):
                        retrieved_docs.append(part.strip())
            
            result["reflective_agent"] = {
                "answer": reflective_result.get("answer", ""),
                "cycles": reflective_result.get("cycles", 1),
                "retrieved_docs": retrieved_docs,
                "confidence": reflective_result.get("score", 0.0),
                "grade": reflective_result.get("grade", ""),
                "feedback": reflective_result.get("feedback", ""),
                "reflection_history": [] 
            }

        if request.method == "both" and result["linear_rag"] and result["reflective_agent"]:
            result["comparison"] = {
                "linear_cycles": result["linear_rag"]["cycles"],
                "reflective_cycles": result["reflective_agent"]["cycles"],
                "cycle_difference": result["reflective_agent"]["cycles"] - result["linear_rag"]["cycles"],
                "linear_confidence": result["linear_rag"]["confidence"],
                "reflective_confidence": result["reflective_agent"]["confidence"],
                "confidence_improvement": result["reflective_agent"]["confidence"] - result["linear_rag"]["confidence"]
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

    result["processing_time"] = time.time() - start_time
    return QueryResponse(**result)


@app.post("/api/query-form")
async def query_form(
    question: str = Form(...),
    method: str = Form("both")
):
    request_obj = QueryRequest(question=question, method=method)
    return await query(request_obj)


@app.get("/api/metrics")
async def get_metrics():
    try:
        results_file = Path(__file__).parent.parent.parent / "results" / "eval_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                eval_data = json.load(f)

            return {
                "evaluation_summary": {
                    "linear_rag": {
                        "avg_f1": eval_data["linear_rag"]["avg_f1"],
                        "avg_cycles": eval_data["linear_rag"]["avg_cycles"]
                    },
                    "reflective_agent": {
                        "avg_f1": eval_data["reflective_agent"]["avg_f1"],
                        "avg_cycles": eval_data["reflective_agent"]["avg_cycles"]
                    },
                    "improvement": eval_data["comparison"]
                },
                "timestamp": eval_data.get("timestamp")
            }
        else:
            return {"message": "No evaluation data available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load metrics: {str(e)}")


@app.get("/api/config")
async def get_config():
    from src.config import (
        MODEL_NAME, EMBEDDING_MODEL, TOP_K, CHUNK_SIZE,
        CHUNK_OVERLAP, MAX_REFLECTION_CYCLES, GRADER_THRESHOLD
    )

    return {
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "retrieval": {
            "top_k": TOP_K,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        },
        "reflection": {
            "max_cycles": MAX_REFLECTION_CYCLES,
            "grader_threshold": GRADER_THRESHOLD
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)