"""
Course Ingestion Module (Silo B)
Ingests course JSONs into a separate vector collection and builds in-memory indexes.
"""
import os
import json
import pickle
import re
from typing import Dict, List, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
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


def extract_instructor(course_data: Dict) -> str:
    """Extract instructor name from various possible keys."""
    # Try different possible keys for instructor
    for key in ['Instructor', 'Instructors', 'Faculty', 'Professor', 'Taught By']:
        if key in course_data and course_data[key]:
            val = course_data[key]
            if isinstance(val, list):
                return ', '.join(str(v) for v in val if v)
            return str(val)
    return ""


def json_to_text(course_data: Dict) -> str:
    """
    Convert course JSON to a rich text representation for semantic search.
    This flattens the JSON into a searchable text document.
    """
    lines = []
    
    # Core fields
    code = course_data.get('Course Code', '')
    name = course_data.get('Course Name', '')
    credits = course_data.get('Credits', '')
    offered_to = course_data.get('Course Offered to', '')
    description = course_data.get('Course Description', '')
    instructor = extract_instructor(course_data)
    
    lines.append(f"Course Code: {code}")
    lines.append(f"Course Name: {name}")
    if credits:
        lines.append(f"Credits: {credits}")
    if offered_to:
        lines.append(f"Offered to: {offered_to}")
    if instructor:
        lines.append(f"Instructor: {instructor}")
    if description:
        lines.append(f"Description: {description}")
    
    # Prerequisites
    prereqs = course_data.get('Prerequisites', {})
    if prereqs:
        if isinstance(prereqs, dict):
            mandatory = prereqs.get('Mandatory', '')
            desirable = prereqs.get('Desirable', '')
            if mandatory:
                if isinstance(mandatory, list):
                    mandatory = ', '.join(str(m) for m in mandatory if m)
                lines.append(f"Mandatory Prerequisites: {mandatory}")
            if desirable:
                if isinstance(desirable, list):
                    desirable = ', '.join(str(d) for d in desirable if d)
                lines.append(f"Desirable Prerequisites: {desirable}")
        elif isinstance(prereqs, str):
            lines.append(f"Prerequisites: {prereqs}")
    
    # Course Outcomes
    outcomes = course_data.get('Course Outcomes', {})
    if outcomes:
        lines.append("Course Outcomes:")
        if isinstance(outcomes, dict):
            for co_key, co_val in outcomes.items():
                lines.append(f"  - {co_key}: {co_val}")
        elif isinstance(outcomes, list):
            for item in outcomes:
                if isinstance(item, dict):
                    for k, v in item.items():
                        lines.append(f"  - {k}: {v}")
                else:
                    lines.append(f"  - {item}")
    
    # Weekly Lecture Plan - extract topics
    weekly_plan = course_data.get('Weekly Lecture Plan', [])
    if weekly_plan:
        topics = []
        for week in weekly_plan:
            if isinstance(week, dict):
                topic = week.get('Lecture Topic', '') or week.get('Topic', '')
                if topic:
                    topics.append(topic)
        if topics:
            lines.append(f"Topics Covered: {'; '.join(topics)}")
    
    # Assessment Plan
    assessment = course_data.get('Assessment Plan', {})
    if assessment:
        lines.append("Assessment Plan:")
        if isinstance(assessment, dict):
            for comp, weight in assessment.items():
                lines.append(f"  - {comp}: {weight}%")
    
    # Resource Material
    resources = course_data.get('Resource Material', {})
    if resources:
        lines.append("Resource Material:")
        if isinstance(resources, dict):
            for rtype, rval in resources.items():
                if rval:
                    lines.append(f"  - {rtype}: {rval}")
    
    return '\n'.join(lines)


def load_course_jsons(jsons_dir: str) -> List[Dict[str, Any]]:
    """
    Load all course JSON files from directory.
    Skips error files (*_error.txt).
    """
    courses = []
    
    if not os.path.exists(jsons_dir):
        print(f"Warning: JSON directory not found: {jsons_dir}")
        return courses
    
    for filename in os.listdir(jsons_dir):
        # Skip error files
        if filename.endswith('_error.txt') or filename.endswith('.txt'):
            continue
        
        if filename.endswith('.json'):
            filepath = os.path.join(jsons_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_source_file'] = filename
                    courses.append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse {filename}: {e}")
            except Exception as e:
                print(f"Warning: Error loading {filename}: {e}")
    
    print(f"Loaded {len(courses)} course JSONs from {jsons_dir}")
    return courses


def build_course_index(courses: List[Dict]) -> Dict[str, Any]:
    """
    Build in-memory index for fast lookups.
    Returns:
        {
            'by_code': {normalized_code: course_data},
            'by_name': {lowercase_name: [course_data, ...]},
            'by_instructor': {lowercase_instructor: [course_data, ...]},
            'all_codes': [list of all course codes],
            'all_names': [list of all course names],
            'all_instructors': [list of unique instructors]
        }
    """
    index = {
        'by_code': {},
        'by_name': {},
        'by_instructor': {},
        'all_codes': [],
        'all_names': [],
        'all_instructors': set()
    }
    
    for course in courses:
        code = normalize_course_code(course.get('Course Code', ''))
        name = course.get('Course Name', '')
        instructor = extract_instructor(course)
        
        # Handle name being a list
        if isinstance(name, list):
            name = name[0] if name else ''
        if not isinstance(name, str):
            name = str(name) if name else ''
        
        # Index by code
        if code:
            index['by_code'][code] = course
            index['all_codes'].append(course.get('Course Code', code))
        
        # Index by name (lowercase for fuzzy matching)
        if name:
            name_lower = name.lower()
            if name_lower not in index['by_name']:
                index['by_name'][name_lower] = []
            index['by_name'][name_lower].append(course)
            index['all_names'].append(name)
        
        # Index by instructor
        if instructor:
            for instr in instructor.split(','):
                instr = instr.strip().lower()
                if instr:
                    if instr not in index['by_instructor']:
                        index['by_instructor'][instr] = []
                    index['by_instructor'][instr].append(course)
                    index['all_instructors'].add(instructor.strip())
    
    index['all_instructors'] = list(index['all_instructors'])
    return index


def ingest_courses(jsons_dir: str = None):
    """
    Main ingestion function for course data (Silo B).
    Creates:
    - Vector Collection B (ChromaDB) for semantic search
    - BM25 Index B for keyword search  
    - In-memory index (pickled) for exact/fuzzy lookups
    - Master list text file
    """
    if jsons_dir is None:
        jsons_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'jsons')
    
    print(f"Starting course ingestion from: {jsons_dir}")
    
    # 1. Load all course JSONs
    courses = load_course_jsons(jsons_dir)
    if not courses:
        print("No courses found. Aborting course ingestion.")
        return
    
    # 2. Build in-memory index
    course_index = build_course_index(courses)
    print(f"Built index: {len(course_index['by_code'])} codes, {len(course_index['by_name'])} names, {len(course_index['all_instructors'])} instructors")
    
    # 3. Convert to Documents for vector/BM25
    documents = []
    for course in courses:
        text_content = json_to_text(course)
        code = normalize_course_code(course.get('Course Code', ''))
        
        # Helper to flatten any list values to strings for ChromaDB metadata
        def flatten_meta(val):
            if isinstance(val, list):
                return ', '.join(str(v) for v in val if v)
            return str(val) if val else ''
        
        doc = Document(
            page_content=text_content,
            metadata={
                'course_code': flatten_meta(course.get('Course Code', '')),
                'course_code_normalized': code,
                'course_name': flatten_meta(course.get('Course Name', '')),
                'credits': flatten_meta(course.get('Credits', '')),
                'offered_to': flatten_meta(course.get('Course Offered to', '')),
                'instructor': extract_instructor(course),
                'source_file': course.get('_source_file', ''),
                'type': 'course'
            }
        )
        documents.append(doc)
    
    print(f"Created {len(documents)} documents for indexing")
    
    # 4. Create output directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 5. Vector Index (ChromaDB Collection B)
    course_chroma_dir = os.path.join(data_dir, 'course_chroma_db')
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
    
    # Delete existing collection if exists
    if os.path.exists(course_chroma_dir):
        import shutil
        shutil.rmtree(course_chroma_dir)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=course_chroma_dir
    )
    print(f"Course vector store created at {course_chroma_dir}")
    
    # 6. BM25 Index B
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_path = os.path.join(data_dir, 'course_bm25_retriever.pkl')
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25_retriever, f)
    print(f"Course BM25 retriever saved to {bm25_path}")
    
    # 7. Save in-memory index
    index_path = os.path.join(data_dir, 'course_index.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump(course_index, f)
    print(f"Course index saved to {index_path}")
    
    # 8. Save raw courses for direct JSON access
    courses_path = os.path.join(data_dir, 'courses_raw.pkl')
    with open(courses_path, 'wb') as f:
        pickle.dump(courses, f)
    print(f"Raw courses saved to {courses_path}")
    
    # 9. Generate Master List text file
    master_list_path = os.path.join(data_dir, 'course_master_list.txt')
    with open(master_list_path, 'w', encoding='utf-8') as f:
        f.write("IIIT Delhi Course Master List\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Courses: {len(courses)}\n\n")
        
        # Group by department prefix
        by_dept = {}
        for course in courses:
            code = course.get('Course Code', 'UNKNOWN')
            # Handle list case
            if isinstance(code, list):
                code = code[0] if code else 'UNKNOWN'
            if not isinstance(code, str):
                code = str(code) if code else 'UNKNOWN'
            # Extract department prefix (e.g., CSE, ECE, BIO)
            dept = re.match(r'^([A-Z]+)', code)
            dept = dept.group(1) if dept else 'OTHER'
            if dept not in by_dept:
                by_dept[dept] = []
            by_dept[dept].append(course)
        
        # Helper to safely get string from potentially list value
        def safe_str(val, default=''):
            if isinstance(val, list):
                return val[0] if val else default
            return str(val) if val else default
        
        for dept in sorted(by_dept.keys()):
            f.write(f"\n## {dept} Courses ({len(by_dept[dept])})\n")
            f.write("-" * 40 + "\n")
            for course in sorted(by_dept[dept], key=lambda c: safe_str(c.get('Course Code', ''))):
                code = safe_str(course.get('Course Code', ''))
                name = safe_str(course.get('Course Name', ''))
                credits = safe_str(course.get('Credits', ''))
                f.write(f"  {code}: {name} ({credits} credits)\n")
    
    print(f"Master list saved to {master_list_path}")
    
    # 10. Summary
    print("\n" + "=" * 60)
    print("COURSE INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Total Courses: {len(courses)}")
    print(f"  Departments: {', '.join(sorted(by_dept.keys()))}")
    print(f"  Vector DB: {course_chroma_dir}")
    print(f"  BM25 Index: {bm25_path}")
    print(f"  Course Index: {index_path}")


if __name__ == "__main__":
    ingest_courses()
