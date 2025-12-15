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

# Try to import course modules
try:
    from core.course_ingestion import ingest_courses
    from core.course_retrieval import CourseRetriever
    COURSE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Course modules not available: {e}")
    COURSE_MODULES_AVAILABLE = False

app = FastAPI(title="IIITD Chatbot Backend - Dual Engine")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
retriever = None
course_retriever = None
pipeline = None

def initialize_pipeline(use_router: bool = True):
    """Initialize the dual-engine pipeline."""
    global retriever, course_retriever, pipeline
    
    # Check if general vector store exists, if not, ingest
    if not os.path.exists(Config.CHROMA_PERSIST_DIRECTORY):
        print("General vector store not found. Ingesting general data...")
        try:
            ingest_data()
        except Exception as e:
            print(f"Error during general ingestion: {e}")
            return

    try:
        # Engine A: General Retriever
        retriever = get_filterable_retriever()
        print("Engine A (General Retriever) initialized.")
        
        # Engine B: Course Retriever (optional)
        course_retriever = None
        if COURSE_MODULES_AVAILABLE:
            course_index_path = os.path.join(
                os.path.dirname(Config.CHROMA_PERSIST_DIRECTORY),
                'course_index.pkl'
            )
            if os.path.exists(course_index_path):
                try:
                    course_retriever = CourseRetriever()
                    print("Engine B (Course Retriever) initialized.")
                except Exception as e:
                    print(f"Warning: Could not initialize Course Retriever: {e}")
            else:
                print("Course index not found. Run course ingestion to enable Engine B.")
        
        # Create pipeline with both engines
        pipeline = RAGPipeline(
            retriever=retriever,
            use_router=use_router,
            course_retriever=course_retriever
        )
        print("Dual-Engine Pipeline initialized successfully.")
        
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        import traceback
        traceback.print_exc()

# Initialize on startup
initialize_pipeline()

class ChatRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = []

class Source(BaseModel):
    content: str
    metadata: dict

class RouteInfo(BaseModel):
    intent: str = "general"  # 'course', 'general', 'greeting', or 'off_topic'
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
                intent=result["route_info"].get("intent", "general"),
                relevant_sections=result["route_info"].get("relevant_sections", []),
                keywords=result["route_info"].get("keywords", []),
                reasoning=result["route_info"].get("reasoning", ""),
                skip_retrieval=result["route_info"].get("skip_retrieval", False)
            )
        
        return ChatResponse(
            answer=result["answer"], 
            sources=result["sources"],
            route_info=route_info
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def trigger_ingestion():
    """Ingest general knowledge base."""
    try:
        ingest_data()
        initialize_pipeline()
        return {"status": "General ingestion successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest-courses")
async def trigger_course_ingestion():
    """Ingest course JSONs into Engine B."""
    if not COURSE_MODULES_AVAILABLE:
        raise HTTPException(status_code=500, detail="Course modules not available")
    
    try:
        # Ingest courses
        jsons_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'jsons')
        ingest_courses(jsons_dir)
        
        # Reinitialize pipeline to pick up new course data
        initialize_pipeline()
        
        return {"status": "Course ingestion successful"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get system status."""
    return {
        "general_retriever": retriever is not None,
        "course_retriever": course_retriever is not None,
        "pipeline": pipeline is not None,
        "course_modules_available": COURSE_MODULES_AVAILABLE
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
