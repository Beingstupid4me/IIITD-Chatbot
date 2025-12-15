from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .config import Config
from .router import SitemapRouter
from .retrieval import FilterableHybridRetriever

# Try to import Google GenAI, but don't fail if not available
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

# Try to import course retriever
try:
    from .course_retrieval import CourseRetriever
    COURSE_RETRIEVER_AVAILABLE = True
except ImportError:
    COURSE_RETRIEVER_AVAILABLE = False
    print("Warning: CourseRetriever not available. Run course ingestion first.")


class RAGPipeline:
    def __init__(self, retriever, use_router: bool = True, course_retriever=None):
        """
        Initialize the RAG pipeline with dual retrieval engines.
        
        Args:
            retriever: FilterableHybridRetriever for general queries (Engine A)
            use_router: Whether to use the LLM-based router for intent classification
            course_retriever: CourseRetriever for course queries (Engine B)
        """
        self.retriever = retriever  # Engine A: General
        self.course_retriever = course_retriever  # Engine B: Course
        self.use_router = use_router
        
        # Determine which LLM to use
        if Config.LOCAL_MODEL_API and Config.LOCAL_MODEL_API.lower() != "null":
            print(f"Using Local Model: {Config.LOCAL_MODEL_NAME} at {Config.LOCAL_MODEL_API}")
            self.llm = ChatOpenAI(
                base_url=Config.LOCAL_MODEL_API,
                model=Config.LOCAL_MODEL_NAME or "local-model",
                api_key="ignore-me",
                temperature=0
            )
        elif Config.GEMINI_API_KEY and GOOGLE_GENAI_AVAILABLE:
            print("Using Gemini Model")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                temperature=0, 
                google_api_key=Config.GEMINI_API_KEY
            )
        else:
            raise ValueError("No valid API Key found. Please set LOCAL_MODEL_API or install langchain-google-genai with GEMINI_API_KEY")
        
        # Initialize router if enabled
        self.router = None
        if self.use_router:
            try:
                self.router = SitemapRouter(llm=self.llm)
                print("Dual Intent Router initialized.")
            except Exception as e:
                print(f"Warning: Could not initialize router: {e}")
                self.router = None
        
        # Condenser
        self.condense_q_chain = self._create_condenser_chain()
        
        # Generator
        self.rag_chain = self._create_rag_chain()

    def _create_condenser_chain(self):
        condense_q_system_prompt = """You are a query rewriter. Your ONLY job is to rewrite the user's latest input into a standalone question.

**STRICT RULES:**
1. Output ONLY the rewritten question - nothing else.
2. Do NOT answer the question.
3. Do NOT add explanations or context about what you found.
4. Do NOT say things like "Based on the documents..." or "The information shows..."
5. If the input is already a clear question, return it as-is.
6. If the input references previous conversation (e.g., "what about that?", "tell me more"), combine it with context from chat history to form a complete question.
7. If the input is a greeting (hi, hello, thanks), return it as-is.

**Examples:**
- Input: "what about its fees?" (after asking about M.Tech) → Output: "What are the fees for M.Tech?"
- Input: "CSE101" → Output: "CSE101"
- Input: "tell me about admissions" → Output: "Tell me about admissions"
- Input: "hi" → Output: "hi"

**WRONG outputs (NEVER do this):**
- "Based on the documents, the fee is..." ❌
- "The information about CSE101 shows..." ❌
- "I don't have information about..." ❌

Just output the standalone question."""
        
        condense_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", condense_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Rewrite this into a standalone question: {question}"),
            ]
        )
        return condense_q_prompt | self.llm | StrOutputParser()

    def _create_rag_chain(self):
        qa_system_prompt = """You are IIITD-CHATBOT, a helpful AI assistant for the IIIT Delhi website.
You were built by Vinayak Agarwal and Akshat Kothari.

**CRITICAL RULES - FOLLOW STRICTLY:**

1. **ONLY use the Context Documents below.** Do NOT use any outside knowledge about colleges, universities, or education in general.

2. **Do NOT guess or invent information.** If the answer is not explicitly stated in the context, say: "Based on the available IIITD documents, I don't have specific information about that."

3. **IIIT Delhi DOES NOT have Mechanical Engineering, Civil Engineering, or Chemical Engineering.** 
   - The ONLY B.Tech branches at IIITD are: CSE, ECE, CSAM, CSAI, CSD, CSSS, CSB, EVE, and CS+Econ.
   - If asked about branches not in this list, clarify that IIITD does not offer them.

4. **Be precise with names, codes, and numbers.** If a course code or specific detail appears in the context, quote it exactly.

5. **Refuse out-of-scope queries.** Politely decline requests unrelated to IIITD.

6. **Respond in English only.**

**Context Documents:**
{context}

**(Answer based ONLY on the above context. If unsure, say you don't have that information.)**
"""
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                ("human", "{question}"),
            ]
        )

        def format_docs(docs):
            formatted_docs = []
            for i, doc in enumerate(docs):
                formatted_docs.append(f"Document {i+1}:\n{doc.page_content}")
            return "\n\n".join(formatted_docs)

        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def _sanitize_condensed_question(self, condensed: str, original: str) -> str:
        """
        Detect if the condenser accidentally generated an answer instead of a question.
        If so, fall back to the original question.
        """
        # Red flags that indicate the LLM answered instead of rephrasing
        answer_indicators = [
            "based on the",
            "according to",
            "the documents show",
            "the information",
            "i don't have",
            "is not explicitly",
            "not provided",
            "refer to the",
            "contact the",
            "for specific",
            "for precise",
        ]
        
        condensed_lower = condensed.lower()
        
        # If condensed output is much longer than original and contains answer indicators
        if len(condensed) > len(original) * 3:
            for indicator in answer_indicators:
                if indicator in condensed_lower:
                    print(f"  [Condenser] Detected answer output, falling back to original question")
                    return original
        
        # If it starts with phrases that indicate an answer
        if condensed_lower.startswith(("based on", "according to", "the ", "i ", "there ", "this ", "it ")):
            if len(condensed) > 200:  # Answers tend to be long
                print(f"  [Condenser] Detected answer-like output, falling back to original question")
                return original
        
        return condensed

    def run(self, question: str, chat_history: list = []):
        # 1. Condense
        if chat_history:
            raw_condensed = self.condense_q_chain.invoke({"question": question, "chat_history": chat_history})
            standalone_question = self._sanitize_condensed_question(raw_condensed, question)
        else:
            standalone_question = question
        
        print(f"\n{'='*60}")
        print(f"Standalone Question: {standalone_question}")

        # 2. Route to determine intent
        route_info = None
        intent = "general"  # default
        
        if self.router:
            try:
                route_info = self.router.route(standalone_question)
                intent = route_info.get('intent', 'general')
                
                print(f"Router Output:")
                print(f"  - Intent: {intent}")
                print(f"  - Sections: {route_info.get('relevant_sections', [])}")
                print(f"  - Keywords: {route_info.get('keywords', [])}")
                print(f"  - Reasoning: {route_info.get('reasoning', '')}")
                print(f"  - Skip Retrieval: {route_info.get('skip_retrieval', False)}")
                
                # Handle greeting/off-topic queries without retrieval
                if route_info.get('skip_retrieval'):
                    if intent == 'greeting':
                        response = self._handle_greeting(standalone_question)
                    else:  # off_topic
                        response = self._handle_off_topic(standalone_question)
                    
                    return {
                        "answer": response,
                        "sources": [],
                        "route_info": route_info
                    }
            except Exception as e:
                print(f"Router error: {e}")
                intent = "general"

        # 3. Dispatch to appropriate engine based on intent
        if intent == "course" and self.course_retriever:
            return self._run_course_engine(standalone_question, route_info)
        else:
            return self._run_general_engine(standalone_question, route_info)

    def _run_course_engine(self, question: str, route_info: dict):
        """
        Engine B: Course Retriever (Waterfall).
        Used for course-specific queries.
        """
        print(f"\n[Engine B: Course Retriever]")
        
        # Use waterfall retrieval
        courses, tier_used = self.course_retriever.retrieve(question, top_k=5)
        
        print(f"  Retrieved {len(courses)} courses via {tier_used}")
        
        if not courses:
            return {
                "answer": "I couldn't find any courses matching your query. Try specifying a course code (like CSE101) or course name.",
                "sources": [],
                "route_info": route_info
            }
        
        # Format courses for LLM
        context = self._format_courses_for_context(courses)
        
        # Generate response with course-specific prompt
        response = self._generate_course_response(question, context)
        
        # Format sources
        sources = []
        for course in courses:
            sources.append({
                "content": f"{course.get('Course Code', '')}: {course.get('Course Name', '')}",
                "metadata": {
                    "course_code": course.get('Course Code', ''),
                    "course_name": course.get('Course Name', ''),
                    "credits": course.get('Credits', ''),
                    "tier_used": tier_used,
                    "type": "course"
                }
            })
        
        return {
            "answer": response,
            "sources": sources,
            "route_info": route_info
        }

    def _run_general_engine(self, question: str, route_info: dict):
        """
        Engine A: General Retriever (3-source RAG).
        Used for general IIITD queries.
        """
        print(f"\n[Engine A: General Retriever]")
        
        active_retriever = self.retriever
        
        # Apply filters if we have route info
        if route_info and hasattr(self.retriever, 'with_filter'):
            if route_info.get('chroma_filter') or route_info.get('keywords'):
                active_retriever = self.retriever.with_filter(
                    chroma_filter=route_info.get('chroma_filter'),
                    keywords=route_info.get('keywords')
                )
                print(f"  Filter Applied: {route_info.get('chroma_filter')}")

        # Retrieve
        docs = active_retriever.invoke(question)
        print(f"  Retrieved {len(docs)} chunks")

        # Generate
        context = self._format_docs_for_context(docs)
        response = self._generate_general_response(question, context)
        
        return {
            "answer": response,
            "sources": [
                {"content": doc.page_content, "metadata": doc.metadata} 
                for doc in docs
            ],
            "route_info": route_info
        }

    def _format_courses_for_context(self, courses: list) -> str:
        """Format course JSONs into readable context for LLM."""
        formatted = []
        for i, course in enumerate(courses):
            lines = [f"=== Course {i+1} ==="]
            lines.append(f"Code: {course.get('Course Code', 'N/A')}")
            lines.append(f"Name: {course.get('Course Name', 'N/A')}")
            lines.append(f"Credits: {course.get('Credits', 'N/A')}")
            lines.append(f"Offered to: {course.get('Course Offered to', 'N/A')}")
            
            if course.get('Course Description'):
                lines.append(f"Description: {course['Course Description']}")
            
            # Prerequisites
            prereqs = course.get('Prerequisites', {})
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
            outcomes = course.get('Course Outcomes', {})
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
            
            # Topics from weekly plan
            weekly_plan = course.get('Weekly Lecture Plan', [])
            if weekly_plan:
                topics = []
                for week in weekly_plan[:8]:  # First 8 weeks
                    if isinstance(week, dict):
                        topic = week.get('Lecture Topic', '') or week.get('Topic', '')
                        if topic:
                            topics.append(topic)
                if topics:
                    lines.append(f"Topics: {'; '.join(topics)}")
            
            # Assessment
            assessment = course.get('Assessment Plan', {})
            if assessment and isinstance(assessment, dict):
                assessment_str = ', '.join(f"{k}: {v}%" for k, v in assessment.items())
                lines.append(f"Assessment: {assessment_str}")
            
            # Resources
            resources = course.get('Resource Material', {})
            if resources and isinstance(resources, dict):
                textbook = resources.get('Textbook', '')
                if textbook:
                    lines.append(f"Textbook: {textbook}")
            
            formatted.append('\n'.join(lines))
        
        return '\n\n'.join(formatted)

    def _format_docs_for_context(self, docs: list) -> str:
        """Format retrieved documents into context string."""
        formatted = []
        for i, doc in enumerate(docs):
            formatted.append(f"Document {i+1}:\n{doc.page_content}")
        return "\n\n".join(formatted)

    def _generate_course_response(self, question: str, context: str) -> str:
        """Generate response for course queries."""
        course_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are IIITD-CHATBOT, an AI assistant specializing in IIIT Delhi course information.
You were built by Vinayak Agarwal and Akshat Kothari.

**CRITICAL RULES:**

1. **ONLY use the Course Information below.** Do NOT invent courses or details.

2. **Format nicely.** Present course information in a clean, readable format:
   - Use bullet points for lists
   - Highlight course codes
   - Group related information

3. **Do NOT output raw JSON.** Convert the structured data into natural language.

4. **If a specific detail is not in the context, say so.** Don't make up prerequisites, credits, or instructors.

5. **For "list all" queries:** Present a clean list of matching courses with their codes and names.

6. **Be concise but complete.** Include relevant details like prerequisites, credits, and key topics.

**Course Information:**
{context}

Answer the user's question based on the course information above:"""),
            ("human", "{question}")
        ])
        
        chain = course_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def _generate_general_response(self, question: str, context: str) -> str:
        """Generate response for general IIITD queries."""
        qa_system_prompt = """You are IIITD-CHATBOT, a helpful AI assistant for the IIIT Delhi website.
You were built by Vinayak Agarwal and Akshat Kothari.

**CRITICAL RULES - FOLLOW STRICTLY:**

1. **ONLY use the Context Documents below.** Do NOT use any outside knowledge about colleges, universities, or education in general.

2. **Do NOT guess or invent information.** If the answer is not explicitly stated in the context, say: "Based on the available IIITD documents, I don't have specific information about that."

3. **IIIT Delhi DOES NOT have Mechanical Engineering, Civil Engineering, or Chemical Engineering.** 
   - The ONLY B.Tech branches at IIITD are: CSE, ECE, CSAM, CSAI, CSD, CSSS, CSB, EVE, and CS+Econ.
   - If asked about branches not in this list, clarify that IIITD does not offer them.

4. **Be precise with names, codes, and numbers.** If a course code or specific detail appears in the context, quote it exactly.

5. **Refuse out-of-scope queries.** Politely decline requests unrelated to IIITD.

6. **Respond in English only.**

**Context Documents:**
{context}

**(Answer based ONLY on the above context. If unsure, say you don't have that information.)**
"""
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            ("human", "{question}")
        ])
        
        chain = qa_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def _handle_greeting(self, query: str) -> str:
        """Generate a friendly response for greetings without RAG."""
        greeting_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are IIITD-CHATBOT, a friendly AI assistant for IIIT Delhi.
You were built by Vinayak Agarwal and Akshat Kothari.

The user has sent a greeting or casual message. Respond warmly and briefly.
- Introduce yourself as IIITD-CHATBOT if it's a first greeting.
- Offer to help with questions about IIITD (academics, campus life, admissions, etc.).
- Keep it short (1-2 sentences max)."""),
            ("human", "{query}")
        ])
        chain = greeting_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})

    def _handle_off_topic(self, query: str) -> str:
        """Generate a polite redirect for off-topic queries."""
        return "I'm IIITD-CHATBOT, designed to help with questions about IIIT Delhi. I can assist you with information about academics, admissions, campus facilities, student life, and more. How can I help you with IIITD-related queries?"
