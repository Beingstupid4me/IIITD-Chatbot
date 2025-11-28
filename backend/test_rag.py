import os
import sys

# Add the current directory to sys.path to allow imports from core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.retrieval import get_retriever
from core.generation import RAGPipeline
from core.config import Config

def test_rag():
    print("Initializing RAG Pipeline for Testing...")
    
    try:
        retriever = get_retriever()
        pipeline = RAGPipeline(retriever)
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    test_questions = [
        # "What is the attendance policy?",
        # "Who is the HOD of CSE?",
        # "Tell me about the 'One Student One Job' rule.",
        # "Is there a gym on campus?",
        # "What are the research centers at IIITD?",
        "Who is Ranjan Bose?",
        "Why is iiitd a good school for my hild ? is it oka",
        "Is IIITD a state universaty?",
        "How many semesters are there in a ear?"
    ]

    print("\n--- Starting Tests ---\n")

    for q in test_questions:
        print(f"Question: {q}")
        try:
            result = pipeline.run(q)
            print(f"Answer: {result['answer']}\n")
            print("Sources:")
            for i, source in enumerate(result['sources']):
                print(f"  {i+1}. {source['content'][:100]}...")
            print("-" * 50 + "\n")
        except Exception as e:
            print(f"Error processing question '{q}': {e}\n")

if __name__ == "__main__":
    test_rag()
