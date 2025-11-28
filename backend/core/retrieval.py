import os
import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
from .config import Config

class CustomHybridRetriever(BaseRetriever):
    vector_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    reranker: HuggingFaceCrossEncoder
    top_k_retrieval: int
    top_k_rerank: int

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1. Retrieve from both sources
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # 2. RRF Fusion
        # Combine and deduplicate based on content
        all_docs = {}
        
        def apply_rrf(docs):
            for rank, doc in enumerate(docs):
                if doc.page_content not in all_docs:
                    all_docs[doc.page_content] = {"doc": doc, "score": 0.0}
                # RRF score: 1 / (k + rank)
                # Standard RRF constant k=60
                all_docs[doc.page_content]["score"] += 1.0 / (60 + rank + 1)

        apply_rrf(vector_docs)
        apply_rrf(bm25_docs)

        # Sort by RRF score
        sorted_docs = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        candidates = [item["doc"] for item in sorted_docs][:self.top_k_retrieval * 2] # Take top 2*k for reranking

        if not candidates:
            return []

        # 3. Rerank
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in candidates]
        # HuggingFaceCrossEncoder wraps sentence-transformers CrossEncoder
        # We use the score method which is the public API
        scores = self.reranker.score(pairs)
        
        # Attach scores and sort
        scored_docs = []
        for i, doc in enumerate(candidates):
            scored_docs.append((doc, scores[i]))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K
        final_docs = [doc for doc, score in scored_docs[:self.top_k_rerank]]
        return final_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

def get_retriever():
    # 1. Load Vector Store
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": Config.TOP_K_RETRIEVAL})

    # 2. Load BM25 Retriever
    bm25_path = os.path.join(os.path.dirname(Config.CHROMA_PERSIST_DIRECTORY), "bm25_retriever.pkl")
    if not os.path.exists(bm25_path):
        raise FileNotFoundError("BM25 retriever not found. Run ingestion first.")
    
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = Config.TOP_K_RETRIEVAL

    # 3. Initialize Reranker
    reranker = HuggingFaceCrossEncoder(model_name=Config.RERANKER_MODEL_NAME)

    # 4. Return Custom Retriever
    return CustomHybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        top_k_retrieval=Config.TOP_K_RETRIEVAL,
        top_k_rerank=Config.TOP_K_RERANK
    )
