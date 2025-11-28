from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .config import Config

class RAGPipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        
        # Determine which LLM to use
        if Config.LOCAL_MODEL_API and Config.LOCAL_MODEL_API.lower() != "null":
            print(f"Using Local Model: {Config.LOCAL_MODEL_NAME} at {Config.LOCAL_MODEL_API}")
            self.llm = ChatOpenAI(
                base_url=Config.LOCAL_MODEL_API,
                model=Config.LOCAL_MODEL_NAME or "local-model",
                api_key="ignore-me", # Local models usually don't need a real key
                temperature=0
            )
        elif Config.GEMINI_API_KEY:
            print("Using Gemini Model")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                temperature=0, 
                google_api_key=Config.GEMINI_API_KEY
            )
        else:
            raise ValueError("No valid API Key found. Please set GEMINI_API_KEY or LOCAL_MODEL_API in .env")
        
        # Condenser
        self.condense_q_chain = self._create_condenser_chain()
        
        # Generator
        self.rag_chain = self._create_rag_chain()

    def _create_condenser_chain(self):
        condense_q_system_prompt = """Given the following conversation and a follow up input, rephrase the follow up input to be a standalone question or statement that captures the user's full intent, in English.

The user's input might be "fuzzy", vague, or refer implicitly to previous turns. 
Your goal is to output a gramatically corrct and clear, self-contained query that can be used to search a knowledge base.

Guidelines:
1. If the input is a follow-up (e.g., "what about that?", "explain more"), combine it with the relevant context from the chat history to make it specific.
2. If the input is already standalone, keep it mostly as is, but ensure it's clear.
3. If the input is a greeting or phatic communication (e.g., "hi", "thanks"), just return it as is.
4. Do NOT answer the question. Just rephrase it."""
        
        condense_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", condense_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        return condense_q_prompt | self.llm | StrOutputParser()

    def _create_rag_chain(self):
        qa_system_prompt = """You are IIITD-CHATBOT, a helpful AI assistant for the IIITD college website.
Your persona is friendly, knowledgeable about IIITD based only on the provided context, and strictly focused on assisting with IIITD-related queries.
You were built by Vinayak Agarwal and Akshat Kothari.

**Instructions & Guardrails:**
1.  **Respond in English Only:** All parts of your response, including any internal thought processes or reasoning steps, MUST be in English.
2.  **Prioritize Context:** Base your answers *exclusively* on the provided "Context Documents" below.
3.  **Acknowledge Limits:** If the context does not contain the answer, or is empty/irrelevant, clearly state "Based on the available IIITD documents, I don't have specific information about that." Do not try to answer from general knowledge if context is missing or irrelevant.
4.  **Refuse Out-of-Scope:** Politely decline for requests outside IIITD website documents.
5.  **Goal:** Provide accurate, concise answers based *only* on the context. If the context doesn't directly answer, say so.

**Context Documents:**
{context}

**(Answer the Question based on the Context Documents)**
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

    def run(self, question: str, chat_history: list = []):
        # 1. Condense
        if chat_history:
            standalone_question = self.condense_q_chain.invoke({"question": question, "chat_history": chat_history})
        else:
            standalone_question = question
        
        print(f"Standalone Question: {standalone_question}")

        # 2. Retrieve
        docs = self.retriever.invoke(standalone_question)
        print(f"Retrieved {len(docs)} chunks:")
        for i, doc in enumerate(docs):
            print(f"Chunk {i+1}: {doc.page_content[:100]}...") # Print first 100 chars of each chunk

        # 3. Generate
        # We need to pass the retrieved docs to the chain manually since we already retrieved them
        # Re-creating the chain input structure
        
        qa_system_prompt = """You are IIITD-CHATBOT, a helpful AI assistant for the IIITD college website.
Your persona is friendly, knowledgeable about IIITD based only on the provided context, and strictly focused on assisting with IIITD-related queries.
You were built by Vinayak Agarwal and Akshat Kothari.

**Instructions & Guardrails:**
1.  **Respond in English Only:** All parts of your response, including any internal thought processes or reasoning steps, MUST be in English.
2.  **Prioritize Context:** Base your answers *exclusively* on the provided "Context Documents" below.
3.  **Acknowledge Limits:** If the context does not contain the answer, or is empty/irrelevant, clearly state "Based on the available IIITD documents, I don't have specific information about that." Do not try to answer from general knowledge if context is missing or irrelevant.
4.  **Refuse Out-of-Scope:** Politely decline for requests outside IIITD website documents.
5.  **Goal:** Provide accurate, concise answers based *only* on the context. If the context doesn't directly answer, say so.

**Context Documents:**
{context}

**(Answer the Question based on the Context Documents)**
"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                ("human", "{question}"),
            ]
        )
        
        def format_docs_list(docs_list):
            formatted = []
            for i, doc in enumerate(docs_list):
                formatted.append(f"Document {i+1}:\n{doc.page_content}")
            return "\n\n".join(formatted)

        final_chain = (
            qa_prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = final_chain.invoke({
            "context": format_docs_list(docs),
            "question": standalone_question
        })
        
        return {
            "answer": response,
            "sources": [
                {"content": doc.page_content, "metadata": doc.metadata} 
                for doc in docs
            ]
        }
