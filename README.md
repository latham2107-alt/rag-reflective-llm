# rag-reflective-iisc

**Linear RAG vs Reflective Agent for Academic QA**  
LangGraph + Phi-3 Mini | CCE IISc LLMs: A Hands-on Approach | Course Project

---

## Problem Statement

Many LLM-based RAG systems are **linear** (retrieve -> generate -> return), so they cannot refine their own answers when they contain errors or hallucinations. For academic or research-style writing, hallucinated citations or misattributed facts break trust. A **self-reflective agent** that can critique its own output and correct it is closer to real-world assistant quality.

---

## Architecture Overview

```
                        +----------------+
                        |    Question    |
                        +-------+--------+
                                |
              +-----------------+-----------------+
              |                                   |
    +---------v---------+             +-----------v-----------+
    |   LINEAR RAG      |             |  REFLECTIVE AGENT     |
    |                   |             |  (LangGraph)          |
    +---------+---------+             +-----------+-----------+
              |                                   |
    +---------v---------+             +-----------v-----------+
    |    Retrieve       |             |    Retrieve           |
    |    (top-k chunks) |             |    (top-k chunks)     |
    +---------+---------+             +-----------+-----------+
              |                                   |
    +---------v---------+             +-----------v-----------+
    |    Generate       |             |    Generate           |
    |    (single pass)  |             |    (answer node)      |
    +---------+---------+             +-----------+-----------+
              |                                   |
    +---------v---------+                         |
    |    Return Answer  |                         v
    +---------+---------+             +-----------+-----------+
                                      |    Grade Node         |
                                      |  (LLM-as-a-judge)     |
                                      +-----------+-----------+
                                                |
                                     +----------+----------+
                                     |  supported?         |
                                     +----------+----------+
                                        |             |
                                       YES            NO (max 3 cycles)
                                        |             |
                                        v      +------v------+
                                 Return  |  Regenerate Node  |
                                 Answer  |  (focused rewrite)|
                                         +-------------------+
```

---

## Core Idea

We build two parallel systems for the same academic QA task:

1. **Linear RAG Assistant** - Retrieves relevant paper snippets from a small corpus and generates a short summary or Q&A answer in a single pass.

2. **Reflective Agent (LangGraph)** - Incorporates the same retrieval step but adds a **self-grade node**. This node checks if claims are supported by the retrieved text and if citations are consistent. If not, the agent re-generates with focused prompts for up to 2-3 cycles.

---

## Key Questions

- How much does reflection reduce hallucination rate?
- How often does the grader correctly catch bad answers?
- How many cycles are actually needed vs wasted?
- What is the latency and resource overhead per query?

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Microsoft Phi-3-mini (3.8B) via Ollama |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB (local) |
| Agent Framework | LangGraph + LangChain |
| Web Framework | FastAPI + Uvicorn |
| Frontend | HTML5 + CSS3 + Vanilla JavaScript |
| Language | Python 3.10+ |

---

## Installation
I have run this project on my personal laptop.
Response generation is depends on system configuration.

### 1. Clone the repository

```bash
git clone https://github.com/latham2107-alt/rag-reflective-llm.git
cd rag-reflective-llm
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama and pull Phi-3 Mini

Download Ollama from [https://ollama.com](https://ollama.com)

```bash
# Pull the Phi-3 Mini model (3.8B parameters)
ollama pull phi3:mini

# Quick sanity test
ollama run phi3:mini "Hello, how are you?"
```

---

## Project Structure

```
rag-reflective-iisc/
├── README.md
├── requirements.txt
├── run.py                 # Main CLI entry point with web server
├── data/
│   ├── corpus.jsonl       # Academic documents corpus
│   ├── qa.jsonl          # Hand-crafted QA pairs
│   └── chroma_db/        # Vector index (auto-generated)
├── src/
│   ├── __init__.py
│   ├── config.py         # Global config (paths, model names)
│   ├── api/              # NEW: Web interface and API
│   │   ├── __init__.py
│   │   ├── app.py        # FastAPI application
│   │   └── templates/    # HTML templates
│   │       └── index.html # Main web interface
│   ├── indexing/
│   │   ├── __init__.py
│   │   └── build_index.py # Build Chroma index from corpus
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py  # Document retrieval logic
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── linear_rag.py # Linear RAG pipeline
│   │   └── reflective_agent.py # LangGraph reflective agent
│   └── eval/
│       ├── __init__.py
│       └── run_eval.py   # Full evaluation runner
├── results/
│   └── eval_results.json # Evaluation results and metrics
```

---

## Quick Start

### Step 1: Build the vector index

Place your academic documents (`.txt` files) in `data/corpus/`, then run:

```bash
python -m src.indexing.build_index
```

This chunks documents, embeds them, and stores in ChromaDB at `data/chroma_db/`.

### Step 2: Test Linear RAG

```bash
python -m demo.cli_demo
```

This runs a side-by-side comparison of Linear RAG vs Reflective Agent on a sample question.

### Step 3: Run Full Evaluation

```bash
python -m src.eval.run_eval
```

This loads QA pairs from `data/qa.jsonl` and computes accuracy metrics for both systems.

### Step 4: Launch Web Interface (NEW!)

For an interactive web-based experience:

```bash
python run.py web  # or ./.venv/bin/python run.py web --port 8000
```

This starts a web server at `http://localhost:8000` with:
- **Interactive Web UI**: Ask questions and compare both methods side-by-side
- **REST API**: Programmatic access with endpoints like `/api/query`
- **Real-time Metrics**: View system performance and evaluation results
- **API Documentation**: Auto-generated docs at `/docs`

**Web Interface Features:**
-  **Method Comparison**: Choose Linear RAG, Reflective Agent, or both
-  **Performance Metrics**: View F1 scores, processing times, and improvements
-  **Cycle Tracking**: See how many reflection cycles each method uses
-  **Modern UI**: Responsive design with real-time results

---

## How It Works

### Linear RAG Pipeline

1. **Retrieve**: Embed the question, fetch top-k (k=5) chunks from ChromaDB.
2. **Generate**: Pass context + question to Phi-3 Mini with a grounding prompt.
3. **Return**: Output the answer in a single pass.

### Reflective Agent (LangGraph)

The graph has the following nodes:

- **retrieve_node**: Same retriever as linear RAG.
- **answer_node**: Generates an initial answer using the context.
- **grade_node**: LLM-as-a-judge that evaluates if claims are supported by context.
- **regenerate_node**: Re-writes the answer based on grader feedback (max 3 cycles).

The grader outputs JSON with:
- `supported` (bool): Are all claims backed by context?
- `missing_info`: What is unsupported?
- `citation_issues`: Any fake/misleading citations?
- `should_regenerate` (bool): Should we try again?
- `hint`: Advice for fixing the answer.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | 1 if normalized prediction equals ground truth |
| **Token-level F1** | Overlap between prediction and ground truth tokens |
| **Hallucination Rate** | % of answers with unsupported claims |
| **Retention Rate** | % of queries where final answer == first answer |
| **Fix Rate** | % of initially wrong answers corrected after reflection |
| **Avg Cycles** | Mean reflection iterations per query |
| **Latency** | Wall-clock time per query |

---

## Related Work

- **Self-RAG** (Asai et al., ICLR 2024): Learns retrieve-generate-critique via reflection tokens. We implement self-critique at the workflow level using LangGraph instead of training a model.
- **RAGAS**: Framework for RAG evaluation including faithfulness and hallucination metrics.
- **LangGraph**: Stateful, multi-actor LLM application framework from LangChain.

---


