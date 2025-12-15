from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict, Any
from core.retrieval import get_retriever, get_filterable_retriever
from core.generation import RAGPipeline
from core.ingestion import ingest_data
import os
from core.config import Config
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="IIITD Chatbot Backend")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables
retriever = None
pipeline = None

def initialize_pipeline(use_router: bool = True):
    global retriever, pipeline
    # Check if vector store exists, if not, ingest
    if not os.path.exists(Config.CHROMA_PERSIST_DIRECTORY):
        print("Vector store not found. Ingesting data...")
        try:
            ingest_data()
        except Exception as e:
            print(f"Error during initial ingestion: {e}")
            return

    try:
        # Use filterable retriever for router support
        retriever = get_filterable_retriever()
        pipeline = RAGPipeline(retriever, use_router=use_router)
        print("Pipeline initialized successfully (with Sitemap Router).")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")

# Initialize on startup
initialize_pipeline()

class ChatRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = [] # List of (human, ai) tuples

class Source(BaseModel):
    content: str
    metadata: dict

class RouteInfo(BaseModel):
    query_type: str = "rag"  # 'rag', 'greeting', or 'off_topic'
    relevant_sections: List[str] = []
    keywords: List[str] = []
    reasoning: str = ""
    skip_retrieval: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    route_info: Optional[RouteInfo] = None

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global pipeline
    if not pipeline:
        # Try to initialize again
        initialize_pipeline()
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized. Check logs.")
    
    history_messages = []
    for human, ai in request.chat_history:
        history_messages.append(HumanMessage(content=human))
        history_messages.append(AIMessage(content=ai))
    
    try:
        result = pipeline.run(request.question, chat_history=history_messages)
        
        # Parse route_info if available
        route_info = None
        if result.get("route_info"):
            route_info = RouteInfo(
                relevant_sections=result["route_info"].get("relevant_sections", []),
                keywords=result["route_info"].get("keywords", []),
                reasoning=result["route_info"].get("reasoning", "")
            )
        
        return ChatResponse(
            answer=result["answer"], 
            sources=result["sources"],
            route_info=route_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def trigger_ingestion():
    try:
        ingest_data()
        initialize_pipeline()
        return {"status": "Ingestion successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
