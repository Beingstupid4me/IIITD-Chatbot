"""
Course Ingestion Script
Run this to ingest course JSONs into the Course Retriever (Engine B).
"""
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from core.course_ingestion import ingest_courses


if __name__ == "__main__":
    # Default to jsons folder in parent directory
    jsons_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'jsons')
    
    # Allow override via command line
    if len(sys.argv) > 1:
        jsons_dir = sys.argv[1]
    
    print(f"Ingesting courses from: {jsons_dir}")
    ingest_courses(jsons_dir)
    print("\nDone! You can now restart the backend server to use the course retriever.")
