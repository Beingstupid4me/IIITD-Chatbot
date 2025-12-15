import os
import json
import re
import pickle
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from .config import Config


def clean_header(header: str) -> str:
    """
    Strip markdown syntax from headers.
    Removes: **, *, #, trailing whitespace
    Example: 'Section 2: ... Facilities**' -> 'Section 2: ... Facilities'
    """
    if not header:
        return header
    # Remove markdown bold/italic markers
    header = re.sub(r'\*+', '', header)
    # Remove any remaining # markers
    header = re.sub(r'^#+\s*', '', header)
    # Strip whitespace
    return header.strip()


def generate_sitemap(documents: list) -> dict:
    """
    Generate a hierarchical sitemap from the documents.
    Returns a dict with sections and entities for LLM routing.
    """
    # Track unique headers
    sections_map = {}  # Header 1 -> {subsections: set of Header 2}
    
    # Entity extraction (simple keyword-based)
    entities = {
        "departments": set(),
        "programs": set(),
        "facilities": set(),
        "policies": set(),
    }
    
    # Known entity patterns
    dept_keywords = ["CSE", "ECE", "CB", "HCD", "SSH", "Math", "Department"]
    program_keywords = ["B.Tech", "M.Tech", "Ph.D", "PhD", "MBA", "BBA"]
    facility_keywords = ["Library", "Hostel", "Mess", "Lab", "Centre", "Center", "Gym", "Sports"]
    
    for doc in documents:
        h1 = clean_header(doc.metadata.get("Header 1", ""))
        h2 = clean_header(doc.metadata.get("Header 2", ""))
        h3 = clean_header(doc.metadata.get("Header 3", ""))
        
        if h1:
            if h1 not in sections_map:
                sections_map[h1] = {"subsections": set(), "sub_subsections": set()}
            if h2:
                sections_map[h1]["subsections"].add(h2)
            if h3:
                sections_map[h1]["sub_subsections"].add(h3)
        
        # Simple entity extraction from content
        content = doc.page_content
        for kw in dept_keywords:
            if kw in content:
                entities["departments"].add(kw)
        for kw in program_keywords:
            if kw in content:
                entities["programs"].add(kw)
        for kw in facility_keywords:
            if kw in content:
                entities["facilities"].add(kw)
    
    # Convert to serializable format
    sitemap = {
        "sections": [],
        "entities": {k: sorted(list(v)) for k, v in entities.items()}
    }
    
    for h1, data in sections_map.items():
        sitemap["sections"].append({
            "header_1": h1,
            "subsections": sorted(list(data["subsections"])),
            "sub_subsections": sorted(list(data["sub_subsections"]))
        })
    
    # Sort sections by header name
    sitemap["sections"].sort(key=lambda x: x["header_1"])
    
    return sitemap


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

    for doc in md_header_splits:
        # Clean markdown syntax from headers in metadata
        if "Header 1" in doc.metadata:
            doc.metadata["Header 1"] = clean_header(doc.metadata["Header 1"])
        if "Header 2" in doc.metadata:
            doc.metadata["Header 2"] = clean_header(doc.metadata["Header 2"])
        if "Header 3" in doc.metadata:
            doc.metadata["Header 3"] = clean_header(doc.metadata["Header 3"])
        
        # Create a context string from cleaned headers
        header_context = " > ".join(filter(None, [
            doc.metadata.get("Header 1"), 
            doc.metadata.get("Header 2"), 
            doc.metadata.get("Header 3")
        ]))
        
        # Prepend to content
        doc.page_content = f"Context: {header_context}\nContent: {doc.page_content}"

    print(f"Split into {len(md_header_splits)} chunks (with Context Injection).")

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

    # 5. Generate and save sitemap for routing
    sitemap = generate_sitemap(md_header_splits)
    sitemap_path = os.path.join(os.path.dirname(Config.CHROMA_PERSIST_DIRECTORY), "sitemap.json")
    with open(sitemap_path, "w", encoding="utf-8") as f:
        json.dump(sitemap, f, indent=2, ensure_ascii=False)
    print(f"Sitemap saved to {sitemap_path}")
    print(f"  - {len(sitemap['sections'])} top-level sections found")

    # 6. Save chunks summary to a text file for inspection
    chunks_info_path = os.path.join(os.path.dirname(Config.CHROMA_PERSIST_DIRECTORY), "chunks_summary.txt")
    with open(chunks_info_path, "w", encoding="utf-8") as f:
        f.write(f"Total Chunks: {len(md_header_splits)}\n")
        f.write("=" * 80 + "\n\n")
        for i, doc in enumerate(md_header_splits):
            f.write(f"Chunk {i+1}:\n")
            f.write(f"  Metadata: {doc.metadata}\n")
            f.write(f"  Content Length: {len(doc.page_content)} chars\n")
            f.write(f"  Content Preview: {doc.page_content[:200]}...\n")
            f.write("-" * 80 + "\n")
    print(f"Chunks summary saved to {chunks_info_path}")

if __name__ == "__main__":
    ingest_data()
