import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

MODEL_NAME = os.getenv("MODEL_NAME", "phi3:mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

MAX_REFLECTION_CYCLES = int(os.getenv("MAX_REFLECTION_CYCLES", "3"))
GRADER_THRESHOLD = float(os.getenv("GRADER_THRESHOLD", "0.85"))

CORPUS_FILE = DATA_DIR / "corpus.jsonl"
QA_FILE = DATA_DIR / "qa.jsonl"

RESULTS_DIR = BASE_DIR / "results"
EVAL_RESULTS_FILE = RESULTS_DIR / "eval_results.json"
