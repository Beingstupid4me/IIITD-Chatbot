import os
import sys

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ingestion import ingest_data

if __name__ == "__main__":
    print("Starting Manual Ingestion Process...")
    try:
        ingest_data()
        print("Ingestion Completed Successfully.")
    except Exception as e:
        print(f"Ingestion Failed: {e}")
