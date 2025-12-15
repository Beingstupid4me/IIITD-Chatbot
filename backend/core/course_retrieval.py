"""
Course Retriever Module (Engine B)
Implements the Waterfall retrieval strategy for course queries.

Waterfall Hierarchy:
  Tier 1 (Exact/Regex): Course code lookup (e.g., "CSE101", "BIO5xx")
  Tier 2 (Fuzzy Name): Fuzzy string matching on course names
  Tier 3 (Instructor): Filter by professor name
  Tier 4 (Semantic+BM25): Vector + keyword search fallback
"""
import os
import re
import pickle
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from .config import Config


def normalize_course_code(code) -> str:
    """Normalize course code: remove spaces, uppercase."""
    if not code:
        return ""
    # Handle list case (some JSONs have code as a list)
    if isinstance(code, list):
        code = code[0] if code else ""
    if not isinstance(code, str):
        code = str(code)
    return re.sub(r'\s+', '', code.upper())


class CourseRetriever:
    """
    Waterfall retriever for course-related queries.
    Tries increasingly fuzzy search strategies until results are found.
    """
    
    def __init__(self):
        """Initialize the course retriever with all indexes."""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
        # Load in-memory index
        index_path = os.path.join(data_dir, 'course_index.pkl')
        if not os.path.exists(index_path):
            raise FileNotFoundError("Course index not found. Run course ingestion first.")
        
        with open(index_path, 'rb') as f:
            self.index = pickle.load(f)
        
        # Load raw courses for direct JSON access
        courses_path = os.path.join(data_dir, 'courses_raw.pkl')
        with open(courses_path, 'rb') as f:
            self.courses = pickle.load(f)
        
        # Load vector store
        course_chroma_dir = os.path.join(data_dir, 'course_chroma_db')
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
        self.vectorstore = Chroma(
            persist_directory=course_chroma_dir,
            embedding_function=embeddings
        )
        
        # Load BM25
        bm25_path = os.path.join(data_dir, 'course_bm25_retriever.pkl')
        with open(bm25_path, 'rb') as f:
            self.bm25_retriever = pickle.load(f)
        
        # Initialize reranker
        self.reranker = HuggingFaceCrossEncoder(model_name=Config.RERANKER_MODEL_NAME)
        
        # Course code pattern for detection
        self.code_pattern = re.compile(
            r'\b([A-Z]{2,4})\s*(\d{3}[A-Z]?)\b',
            re.IGNORECASE
        )
        
        print("CourseRetriever initialized successfully")
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Dict], str]:
        """
        Main retrieval method implementing the waterfall strategy.
        
        Returns:
            Tuple of (list of course dicts, tier_used)
        """
        print(f"\n[CourseRetriever] Query: {query}")
        
        # Tier 1: Exact/Regex Code Match
        courses, tier = self._tier1_code_match(query)
        if courses:
            print(f"  [Tier 1 - Code Match] Found {len(courses)} course(s)")
            return courses[:top_k], "tier1_code"
        
        # Tier 2: Fuzzy Name Match
        courses, tier = self._tier2_fuzzy_name(query)
        if courses:
            print(f"  [Tier 2 - Fuzzy Name] Found {len(courses)} course(s)")
            return courses[:top_k], "tier2_fuzzy"
        
        # Tier 3: Instructor Match
        courses, tier = self._tier3_instructor(query)
        if courses:
            print(f"  [Tier 3 - Instructor] Found {len(courses)} course(s)")
            return courses[:top_k], "tier3_instructor"
        
        # Tier 4: Semantic + BM25 Search
        courses, tier = self._tier4_semantic_bm25(query, top_k)
        print(f"  [Tier 4 - Semantic+BM25] Found {len(courses)} course(s)")
        return courses, "tier4_semantic"
    
    def _tier1_code_match(self, query: str) -> Tuple[List[Dict], str]:
        """
        Tier 1: Extract course codes from query and fetch exact matches.
        Handles patterns like: CSE101, CSE 101, cse101, BIO5xx
        """
        found_courses = []
        
        # Find all potential course codes in query
        matches = self.code_pattern.findall(query)
        
        for dept, num in matches:
            code = normalize_course_code(f"{dept}{num}")
            
            # Check for exact match
            if code in self.index['by_code']:
                found_courses.append(self.index['by_code'][code])
                continue
            
            # Check for wildcard pattern (e.g., BIO5xx means all BIO5xx courses)
            if 'x' in num.lower() or 'X' in num:
                pattern = code.replace('X', r'\d').replace('x', r'\d')
                for stored_code in self.index['by_code']:
                    if re.match(pattern, stored_code):
                        found_courses.append(self.index['by_code'][stored_code])
        
        # Also check if the entire query looks like a course code
        query_normalized = normalize_course_code(query.strip())
        if query_normalized in self.index['by_code']:
            if self.index['by_code'][query_normalized] not in found_courses:
                found_courses.append(self.index['by_code'][query_normalized])
        
        return found_courses, "tier1"
    
    def _tier2_fuzzy_name(self, query: str, threshold: float = 0.6) -> Tuple[List[Dict], str]:
        """
        Tier 2: Fuzzy string matching on course names.
        """
        query_lower = query.lower()
        matches = []
        
        for name, courses in self.index['by_name'].items():
            # Calculate similarity ratio
            ratio = SequenceMatcher(None, query_lower, name).ratio()
            
            # Also check if query is a substring or vice versa
            if query_lower in name or name in query_lower:
                ratio = max(ratio, 0.8)
            
            # Check for word overlap
            query_words = set(query_lower.split())
            name_words = set(name.split())
            if query_words & name_words:  # If any words overlap
                word_overlap = len(query_words & name_words) / max(len(query_words), len(name_words))
                ratio = max(ratio, word_overlap)
            
            if ratio >= threshold:
                for course in courses:
                    matches.append((ratio, course))
        
        # Sort by similarity and return unique courses
        matches.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        unique_courses = []
        for ratio, course in matches:
            code = course.get('Course Code', '')
            if code not in seen:
                seen.add(code)
                unique_courses.append(course)
        
        return unique_courses, "tier2"
    
    def _tier3_instructor(self, query: str, threshold: float = 0.5) -> Tuple[List[Dict], str]:
        """
        Tier 3: Match by instructor/professor name.
        Handles queries like "courses by Dr. X" or "Prof Y's courses"
        """
        query_lower = query.lower()
        
        # Remove common prefixes
        query_clean = re.sub(r'\b(courses?\s*(by|taught by|from|of)|prof\.?|dr\.?|professor)\b', '', query_lower)
        query_clean = query_clean.strip()
        
        if not query_clean:
            return [], "tier3"
        
        matches = []
        
        for instructor, courses in self.index['by_instructor'].items():
            # Check for substring match
            if query_clean in instructor or instructor in query_clean:
                matches.extend(courses)
                continue
            
            # Fuzzy match
            ratio = SequenceMatcher(None, query_clean, instructor).ratio()
            if ratio >= threshold:
                matches.extend(courses)
        
        # Deduplicate
        seen = set()
        unique_courses = []
        for course in matches:
            code = course.get('Course Code', '')
            if code not in seen:
                seen.add(code)
                unique_courses.append(course)
        
        return unique_courses, "tier3"
    
    def _tier4_semantic_bm25(self, query: str, top_k: int = 5) -> Tuple[List[Dict], str]:
        """
        Tier 4: Semantic + BM25 hybrid search with reranking.
        """
        # Vector search
        vector_docs = self.vectorstore.similarity_search(query, k=top_k * 2)
        
        # BM25 search
        self.bm25_retriever.k = top_k * 2
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # RRF Fusion
        all_docs = {}
        
        def apply_rrf(docs, weight=1.0):
            for rank, doc in enumerate(docs):
                key = doc.metadata.get('course_code_normalized', doc.page_content[:100])
                if key not in all_docs:
                    all_docs[key] = {"doc": doc, "score": 0.0}
                all_docs[key]["score"] += weight * (1.0 / (60 + rank + 1))
        
        apply_rrf(vector_docs, weight=1.0)
        apply_rrf(bm25_docs, weight=1.0)
        
        # Sort by RRF score
        sorted_docs = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        candidates = [item["doc"] for item in sorted_docs][:top_k * 2]
        
        if not candidates:
            return [], "tier4"
        
        # Rerank with cross-encoder
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.score(pairs)
        
        scored_docs = [(doc, scores[i]) for i, doc in enumerate(candidates)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Get course data from index
        final_courses = []
        seen = set()
        for doc, score in scored_docs[:top_k]:
            code = normalize_course_code(doc.metadata.get('course_code', ''))
            if code and code in self.index['by_code'] and code not in seen:
                seen.add(code)
                final_courses.append(self.index['by_code'][code])
        
        return final_courses, "tier4"
    
    def get_all_courses_by_dept(self, dept: str) -> List[Dict]:
        """Get all courses for a department prefix (e.g., 'CSE', 'BIO')."""
        dept = dept.upper()
        courses = []
        for code, course in self.index['by_code'].items():
            if code.startswith(dept):
                courses.append(course)
        return sorted(courses, key=lambda c: c.get('Course Code', ''))
    
    def get_all_courses(self) -> List[Dict]:
        """Get all courses."""
        return self.courses
    
    def search_by_keyword(self, keyword: str) -> List[Dict]:
        """Search courses containing a keyword in name or description."""
        keyword_lower = keyword.lower()
        matches = []
        
        for course in self.courses:
            name = course.get('Course Name', '').lower()
            desc = course.get('Course Description', '').lower()
            
            if keyword_lower in name or keyword_lower in desc:
                matches.append(course)
        
        return matches


def get_course_retriever() -> CourseRetriever:
    """Factory function to get a CourseRetriever instance."""
    return CourseRetriever()
