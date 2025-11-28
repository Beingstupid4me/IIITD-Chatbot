import os
import pickle
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from .config import Config

def ingest_data():
    print("Starting ingestion...")
    # 1. Load Data
    if not os.path.exists(Config.KNOWLEDGE_BASE_PATH):
        raise FileNotFoundError(f"Knowledge base file not found at {Config.KNOWLEDGE_BASE_PATH}")
    
    with open(Config.KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # 2. Split Data
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_text)

    print(f"Split into {len(md_header_splits)} chunks.")

    # 3. Vector Index (Chroma)
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
    
    # Initialize Chroma
    # Note: Chroma automatically persists if persist_directory is set
    vectorstore = Chroma.from_documents(
        documents=md_header_splits,
        embedding=embeddings,
        persist_directory=Config.CHROMA_PERSIST_DIRECTORY
    )
    print(f"Vector store created at {Config.CHROMA_PERSIST_DIRECTORY}")

    # 4. Sparse Index (BM25)
    bm25_retriever = BM25Retriever.from_documents(md_header_splits)
    
    # Save BM25 retriever
    bm25_path = os.path.join(os.path.dirname(Config.CHROMA_PERSIST_DIRECTORY), "bm25_retriever.pkl")
    os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"BM25 retriever saved to {bm25_path}")

if __name__ == "__main__":
    ingest_data()
