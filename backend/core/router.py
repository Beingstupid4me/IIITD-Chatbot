"""
Sitemap Router Module
Uses LLM to classify queries and determine relevant sections/filters for retrieval.
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
            ("system", """You are a query classifier and router for the IIITD (IIIT Delhi) Knowledge Base.
Your job is to classify user queries and determine which sections of the knowledge base are most relevant.

{sitemap}

---

Given a user query, you must output a JSON object with these fields:
1. "query_type": One of:
   - "greeting" - for greetings, thanks, casual chat (hi, hello, thanks, bye, how are you)
   - "off_topic" - for questions completely unrelated to IIITD (weather, recipes, jokes, etc.)
   - "rag" - for questions that need information from the IIITD knowledge base
2. "relevant_sections": List of Header 1 section names (1-3 max) from the sitemap. Empty list [] for greeting/off_topic.
3. "keywords": Key terms extracted from the query. Empty list [] for greeting/off_topic.
4. "reasoning": Brief explanation (1 sentence).

**Rules:**
- Use EXACT section names from the sitemap.
- For greetings/off_topic, return empty lists for sections and keywords.
- Output ONLY valid JSON, no extra text.

**Example outputs:**
Query: "hello" -> {{"query_type": "greeting", "relevant_sections": [], "keywords": [], "reasoning": "This is a greeting."}}
Query: "what is the fee structure?" -> {{"query_type": "rag", "relevant_sections": ["Section 20: Academic Regulations"], "keywords": ["fee", "structure"], "reasoning": "Fee information is in academic regulations."}}
"""),
            ("human", "Query: {query}\n\nOutput JSON only:")
        ])
        
        return router_prompt | self.llm | StrOutputParser()
    
    def _is_greeting(self, query: str) -> bool:
        """Fast check for common greetings without LLM."""
        query_lower = query.lower().strip()
        for pattern in GREETING_PATTERNS:
            if re.match(pattern, query_lower, re.IGNORECASE):
                return True
        return False

    def _parse_llm_output(self, output: str) -> Dict[str, Any]:
        """Parse LLM output with fallback handling for malformed JSON."""
        # Try to extract JSON from the output
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
            "query_type": "rag",
            "relevant_sections": [],
            "keywords": [],
            "reasoning": "Fallback parsing"
        }
        
        # Check for greeting/off_topic indicators
        if any(word in output.lower() for word in ["greeting", "hello", "hi", "thanks"]):
            result["query_type"] = "greeting"
        elif "off_topic" in output.lower() or "unrelated" in output.lower():
            result["query_type"] = "off_topic"
        
        return result

    def route(self, query: str) -> Dict[str, Any]:
        """
        Route a query to determine relevant sections and keywords.
        
        Returns:
            Dict with keys: query_type, relevant_sections, keywords, reasoning, chroma_filter, skip_retrieval
        """
        # Fast path: check for common greetings without LLM
        if self._is_greeting(query):
            return {
                "query_type": "greeting",
                "relevant_sections": [],
                "keywords": [],
                "reasoning": "Detected as greeting (fast path)",
                "chroma_filter": None,
                "skip_retrieval": True
            }
        
        try:
            # Call LLM router
            raw_output = self.router_chain.invoke({
                "query": query,
                "sitemap": self.sitemap_text
            })
            
            # Parse with fallback handling
            result = self._parse_llm_output(raw_output)
            
            # Determine if we should skip retrieval
            query_type = result.get("query_type", "rag")
            skip_retrieval = query_type in ["greeting", "off_topic"]
            
            # Build ChromaDB filter from sections (only if not skipping)
            chroma_filter = None
            if not skip_retrieval and result.get("relevant_sections"):
                sections = result["relevant_sections"]
                if len(sections) == 1:
                    chroma_filter = {"Header 1": sections[0]}
                elif len(sections) > 1:
                    chroma_filter = {"$or": [{"Header 1": s} for s in sections]}
            
            return {
                "query_type": query_type,
                "relevant_sections": result.get("relevant_sections", []),
                "keywords": result.get("keywords", []),
                "reasoning": result.get("reasoning", ""),
                "chroma_filter": chroma_filter,
                "skip_retrieval": skip_retrieval
            }
        except Exception as e:
            print(f"Router error: {e}")
            # Fallback: assume RAG is needed
            return {
                "query_type": "rag",
                "relevant_sections": [],
                "keywords": [],
                "reasoning": f"Router failed: {e}",
                "chroma_filter": None,
                "skip_retrieval": False
            }
    
    def get_section_names(self) -> List[str]:
        """Get list of all Header 1 section names."""
        return [s.get("header_1", "") for s in self.sitemap.get("sections", [])]


def get_router() -> SitemapRouter:
    """Factory function to create a SitemapRouter instance."""
    return SitemapRouter()
