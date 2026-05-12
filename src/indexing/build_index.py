import jsonlines
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm
from src.config import DATA_DIR, INDEX_DIR, CORPUS_FILE, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

def load_corpus(corpus_path: Path = CORPUS_FILE) -> list[dict]:
    docs = []
    with jsonlines.open(corpus_path, "r") as reader:
        for item in reader:
            docs.append({
                "id": item.get("id", ""),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "source": item.get("source", "")
            })
    return docs


def create_documents(corpus: list[dict]) -> list[Document]:
    docs = []
    for item in corpus:
        text = f"Title: {item['title']}\n{item['content']}"
        docs.append(Document(
            page_content=text,
            metadata={"id": item["id"], "title": item["title"], "source": item.get("source", "")}
        ))
    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(documents)

def build_vectorstore(documents: list[Document], persist_dir: Path = INDEX_DIR) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    os.makedirs(persist_dir, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_dir)
    )
    return vectorstore


def build_index(corpus_path: Path = CORPUS_FILE, persist_dir: Path = INDEX_DIR) -> Chroma:

    print("Loading corpus...")
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus)} documents.")
    
    print("Creating documents...")
    documents = create_documents(corpus)
    
    print("Splitting documents...")
    chunks = split_documents(documents)
    
    print(f"Created {len(chunks)} chunks.")
    
    print("Building vector store...")
    vectorstore = build_vectorstore(chunks, persist_dir)
    print(f"Vector store persisted at {persist_dir}")
    
    return vectorstore

if __name__ == "__main__":
    build_index()
