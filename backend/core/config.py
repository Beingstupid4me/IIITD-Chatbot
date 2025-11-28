import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    LOCAL_MODEL_API = os.getenv("LOCAL_MODEL_API")
    LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME")
    
    CHROMA_PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
    KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "iiitd_kb_master.md")
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Open source embedding
    RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Open source reranker (if using cross-encoder)
    
    # Retrieval settings
    TOP_K_RETRIEVAL = 30
    TOP_K_RERANK = 15
