"""
Dual Intent Router Module
Routes queries to either:
- INTENT_COURSE: Course-related queries (syllabi, prerequisites, credits, instructors, course codes)
- INTENT_GENERAL: General IIITD queries (admissions, fees, campus life, rules, placements)

Also handles greetings and off-topic queries.
"""
import os
import json
import re
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from .config import Config

# Try to import Google GenAI, but don't fail if not available
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False


# Common greetings and off-topic patterns (checked before LLM call)
GREETING_PATTERNS = [
    r"^(hi|hello|hey|hola|namaste|good\s*(morning|afternoon|evening|night))[\s!.,?]*$",
    r"^(thanks|thank\s*you|thx|ty)[\s!.,?]*$",
    r"^(bye|goodbye|see\s*you|cya)[\s!.,?]*$",
    r"^(ok|okay|sure|yes|no|yep|nope|alright)[\s!.,?]*$",
    r"^(how\s*are\s*you|what'?s\s*up|wassup)[\s!.,?]*$",
]

# Course-related patterns for fast detection
COURSE_CODE_PATTERN = re.compile(
    r'\b([A-Z]{2,4})\s*(\d{3}[A-Z]?)\b',
    re.IGNORECASE
)

COURSE_KEYWORDS = [
    'syllabus', 'syllabi', 'prerequisite', 'prerequisites', 'prereq',
    'credits', 'credit hours', 'course outline', 'course description',
    'lecture plan', 'weekly plan', 'course outcome', 'textbook', 'reference book',
    'taught by', 'instructor', 'professor', 'faculty teaching',
    'elective', 'core course', 'open elective',
    'course code', 'what is cse', 'what is ece', 'what is bio', 'what is mth',
    'courses about', 'courses on', 'courses related to',
]


class RouterOutput(BaseModel):
    """Schema for router output"""
    query_type: str = Field(
        description="Type of query: 'rag' (needs retrieval), 'greeting' (casual/greeting), 'off_topic' (unrelated to IIITD)"
    )
    relevant_sections: List[str] = Field(
        description="List of Header 1 section names that are most relevant to the query. Use exact names from the sitemap."
    )
    keywords: List[str] = Field(
        description="Key entities or keywords extracted from the query for BM25 filtering"
    )
    reasoning: str = Field(
        description="Brief reasoning for why these sections were selected"
    )


class SitemapRouter:
    def __init__(self, llm=None):
        """Initialize the router with an LLM and load the sitemap."""
        # Use provided LLM or create one based on config
        if llm:
            self.llm = llm
        elif Config.LOCAL_MODEL_API and Config.LOCAL_MODEL_API.lower() != "null":
            self.llm = ChatOpenAI(
                base_url=Config.LOCAL_MODEL_API,
                model=Config.LOCAL_MODEL_NAME or "local-model",
                api_key="ignore-me",
                temperature=0
            )
        elif Config.GEMINI_API_KEY and GOOGLE_GENAI_AVAILABLE:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                google_api_key=Config.GEMINI_API_KEY
            )
        else:
            raise ValueError("No valid API Key found for Router LLM. Set LOCAL_MODEL_API or install langchain-google-genai.")
        
        # Load sitemap
        self.sitemap = self._load_sitemap()
        self.sitemap_text = self._format_sitemap_for_prompt()
        
        # Create router chain
        self.router_chain = self._create_router_chain()
    
    def _load_sitemap(self) -> Dict[str, Any]:
        """Load the sitemap JSON file."""
        sitemap_path = os.path.join(
            os.path.dirname(Config.CHROMA_PERSIST_DIRECTORY), 
            "sitemap.json"
        )
        if not os.path.exists(sitemap_path):
            print(f"Warning: Sitemap not found at {sitemap_path}. Run ingestion first.")
            return {"sections": [], "entities": {}}
        
        with open(sitemap_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _format_sitemap_for_prompt(self) -> str:
        """Format sitemap into a readable string for the LLM prompt."""
        if not self.sitemap.get("sections"):
            return "No sitemap available."
        
        lines = ["## Knowledge Base Sitemap (Header 1 Sections):\n"]
        for section in self.sitemap.get("sections", []):
            h1 = section.get("header_1", "Unknown")
            lines.append(f"- **{h1}**")
            
            # Add subsections (Header 2) as a brief preview
            subsections = section.get("subsections", [])
            if subsections:
                preview = ", ".join(subsections[:5])  # First 5 subsections
                if len(subsections) > 5:
                    preview += f", ... ({len(subsections)} total)"
                lines.append(f"  - Subsections: {preview}")
        
        # Add entities map
        if self.sitemap.get("entities"):
            lines.append("\n## Key Entities Map:")
            for entity_type, entities in self.sitemap.get("entities", {}).items():
                if entities:
                    preview = ", ".join(entities[:10])
                    if len(entities) > 10:
                        preview += f", ... ({len(entities)} total)"
                    lines.append(f"- **{entity_type}**: {preview}")
        
        return "\n".join(lines)
    
    def _create_router_chain(self):
        """Create the LLM chain for routing queries."""
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier for IIIT Delhi's dual knowledge base system.

**Your job:** Classify user queries into one of these intents:

1. **"course"** - Questions about SPECIFIC COURSES:
   - Syllabus, prerequisites, credits, lecture topics
   - Course codes (CSE101, BIO213, ECE314, MTH100, etc.)
   - Course instructors/professors
   - Course descriptions, textbooks, outcomes
   - "What courses cover X topic?"
   - "List all CSE/ECE/BIO courses"

2. **"general"** - Questions about IIITD in general:
   - Admissions, fees, scholarships
   - Campus facilities, hostels, mess, library
   - Academic rules, attendance, grading, CGPA
   - Placements, internships, companies
   - Faculty research (NOT course teaching)
   - Student clubs, events, fests
   - Branches offered (CSE, ECE, CSAM, etc.)

3. **"greeting"** - Casual greetings: hi, hello, thanks, bye

4. **"off_topic"** - Completely unrelated to IIITD: weather, recipes, jokes

{sitemap}

---

**Output a JSON object with:**
- "intent": one of ["course", "general", "greeting", "off_topic"]
- "relevant_sections": List of section names (for "general" only, empty for others)
- "keywords": Key terms from query
- "reasoning": Brief explanation (1 sentence)

**Examples:**
- "CSE101 syllabus" → {{"intent": "course", "relevant_sections": [], "keywords": ["CSE101", "syllabus"], "reasoning": "Asking for course syllabus"}}
- "What are the prerequisites for Machine Learning?" → {{"intent": "course", "relevant_sections": [], "keywords": ["prerequisites", "Machine Learning"], "reasoning": "Course prerequisite query"}}
- "Fee structure?" → {{"intent": "general", "relevant_sections": ["Section 20: Academic Eligibility, Regulations & Ordinances"], "keywords": ["fee", "structure"], "reasoning": "General fees query"}}
- "hello" → {{"intent": "greeting", "relevant_sections": [], "keywords": [], "reasoning": "Greeting"}}

Output ONLY valid JSON:"""),
            ("human", "Query: {query}")
        ])
        
        return router_prompt | self.llm | StrOutputParser()
    
    def _is_greeting(self, query: str) -> bool:
        """Fast check for common greetings without LLM."""
        query_lower = query.lower().strip()
        for pattern in GREETING_PATTERNS:
            if re.match(pattern, query_lower, re.IGNORECASE):
                return True
        return False
    
    def _is_course_query(self, query: str) -> bool:
        """Fast check if query is likely course-related."""
        query_lower = query.lower()
        
        # Check for course code pattern
        if COURSE_CODE_PATTERN.search(query):
            return True
        
        # Check for course keywords
        for keyword in COURSE_KEYWORDS:
            if keyword in query_lower:
                return True
        
        return False

    def _parse_llm_output(self, output: str) -> Dict[str, Any]:
        """Parse LLM output with fallback handling for malformed JSON."""
        output = output.strip()
        
        # Try direct JSON parse
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in the output (between { and })
        json_match = re.search(r'\{[^{}]*\}', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON with nested braces
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to extract key information with regex
        result = {
            "intent": "general",
            "relevant_sections": [],
            "keywords": [],
            "reasoning": "Fallback parsing"
        }
        
        # Check for intent indicators
        output_lower = output.lower()
        if any(word in output_lower for word in ["greeting", "hello", "hi ", "thanks"]):
            result["intent"] = "greeting"
        elif "off_topic" in output_lower or "unrelated" in output_lower:
            result["intent"] = "off_topic"
        elif "course" in output_lower and ("syllabus" in output_lower or "prerequisite" in output_lower or "credit" in output_lower):
            result["intent"] = "course"
        
        return result

    def route(self, query: str) -> Dict[str, Any]:
        """
        Route a query to determine intent and relevant filters.
        
        Returns:
            Dict with keys:
            - intent: 'course', 'general', 'greeting', or 'off_topic'
            - relevant_sections: List of section names (for general only)
            - keywords: Key terms from query
            - reasoning: Explanation
            - chroma_filter: Filter for ChromaDB (general only)
            - skip_retrieval: Whether to skip retrieval entirely
        """
        # Fast path: check for common greetings without LLM
        if self._is_greeting(query):
            return {
                "intent": "greeting",
                "relevant_sections": [],
                "keywords": [],
                "reasoning": "Detected as greeting (fast path)",
                "chroma_filter": None,
                "skip_retrieval": True
            }
        
        # Fast path: check for obvious course queries
        is_likely_course = self._is_course_query(query)
        
        try:
            # Call LLM router
            raw_output = self.router_chain.invoke({
                "query": query,
                "sitemap": self.sitemap_text
            })
            
            # Parse with fallback handling
            result = self._parse_llm_output(raw_output)
            
            # Get intent (handle both old 'query_type' and new 'intent' keys)
            intent = result.get("intent") or result.get("query_type", "general")
            
            # Override with fast path detection if LLM missed it
            if is_likely_course and intent == "general":
                intent = "course"
                print(f"  [Router] Overriding to 'course' based on fast path detection")
            
            # Map old values to new
            if intent == "rag":
                intent = "general"
            
            # Determine if we should skip retrieval
            skip_retrieval = intent in ["greeting", "off_topic"]
            
            # Build ChromaDB filter from sections (only for general intent)
            chroma_filter = None
            if intent == "general" and result.get("relevant_sections"):
                sections = result["relevant_sections"]
                if len(sections) == 1:
                    chroma_filter = {"Header 1": sections[0]}
                elif len(sections) > 1:
                    chroma_filter = {"$or": [{"Header 1": s} for s in sections]}
            
            return {
                "intent": intent,
                "relevant_sections": result.get("relevant_sections", []),
                "keywords": result.get("keywords", []),
                "reasoning": result.get("reasoning", ""),
                "chroma_filter": chroma_filter,
                "skip_retrieval": skip_retrieval
            }
        except Exception as e:
            print(f"Router error: {e}")
            # Fallback: use fast path detection or default to general
            fallback_intent = "course" if is_likely_course else "general"
            return {
                "intent": fallback_intent,
                "relevant_sections": [],
                "keywords": [],
                "reasoning": f"Router failed: {e}, using fallback",
                "chroma_filter": None,
                "skip_retrieval": False
            }
    
    def get_section_names(self) -> List[str]:
        """Get list of all Header 1 section names."""
        return [s.get("header_1", "") for s in self.sitemap.get("sections", [])]


def get_router() -> SitemapRouter:
    """Factory function to create a SitemapRouter instance."""
    return SitemapRouter()
