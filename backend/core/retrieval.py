import os
import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional, Dict, Any
from pydantic import Field
from .config import Config


class FilterableHybridRetriever(BaseRetriever):
    """
    A hybrid retriever that combines 3 sources:
    1. BM25 (Keyword) - Exact match anchor for names, codes
    2. Global Vector - Semantic search across ALL documents (no filter)
    3. Scoped Vector (Router) - Filtered search in specific sections
    
    Results are fused using RRF and reranked with a cross-encoder.
    """
    vectorstore: Any = Field(description="ChromaDB vectorstore instance")
    bm25_retriever: Any = Field(description="BM25 retriever instance")
    reranker: HuggingFaceCrossEncoder
    top_k_retrieval: int
    top_k_rerank: int
    
    # Optional filters (for scoped vector search)
    chroma_filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter for scoped vector search")
    keyword_boost: Optional[List[str]] = Field(default=None, description="Keywords to boost in BM25")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        all_docs = {}
        
        def apply_rrf(docs, weight=1.0, source_name=""):
            """Apply RRF scoring to a list of documents."""
            for rank, doc in enumerate(docs):
                if doc.page_content not in all_docs:
                    all_docs[doc.page_content] = {"doc": doc, "score": 0.0, "sources": []}
                # RRF score: weight / (k + rank)
                all_docs[doc.page_content]["score"] += weight * (1.0 / (60 + rank + 1))
                if source_name and source_name not in all_docs[doc.page_content]["sources"]:
                    all_docs[doc.page_content]["sources"].append(source_name)

        # === SOURCE 1: BM25 (Keyword Search) ===
        # The "Exact Match" anchor - catches course codes, names, specific terms
        bm25_query = query
        if self.keyword_boost:
            bm25_query = f"{query} {' '.join(self.keyword_boost)}"
        
        self.bm25_retriever.k = self.top_k_retrieval
        bm25_docs = self.bm25_retriever.invoke(bm25_query)
        apply_rrf(bm25_docs, weight=1.0, source_name="BM25")
        print(f"  [BM25] Retrieved {len(bm25_docs)} docs")

        # === SOURCE 2: Global Vector Search (No Filter) ===
        # The "Vibe" anchor - catches semantic meaning across entire KB
        global_vector_docs = self.vectorstore.similarity_search(
            query, 
            k=self.top_k_retrieval
        )
        apply_rrf(global_vector_docs, weight=1.0, source_name="GlobalVector")
        print(f"  [GlobalVector] Retrieved {len(global_vector_docs)} docs")

        # === SOURCE 3: Scoped Vector Search (With Router Filter) ===
        # The "Specialist" - drills down into specific sections
        if self.chroma_filter:
            scoped_vector_docs = self.vectorstore.similarity_search(
                query, 
                k=self.top_k_retrieval,
                filter=self.chroma_filter
            )
            # Give scoped results slightly higher weight since they're targeted
            apply_rrf(scoped_vector_docs, weight=1.2, source_name="ScopedVector")
            print(f"  [ScopedVector] Retrieved {len(scoped_vector_docs)} docs with filter: {self.chroma_filter}")
        else:
            print(f"  [ScopedVector] Skipped (no filter provided)")

        # === RRF Fusion ===
        # Sort by combined RRF score
        sorted_docs = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        candidates = [item["doc"] for item in sorted_docs][:self.top_k_retrieval * 2]
        
        print(f"  [RRF Fusion] {len(all_docs)} unique docs -> {len(candidates)} candidates for reranking")

        if not candidates:
            return []

        # === Rerank with Cross-Encoder ===
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.score(pairs)
        
        scored_docs = [(doc, scores[i]) for i, doc in enumerate(candidates)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        final_docs = [doc for doc, score in scored_docs[:self.top_k_rerank]]
        print(f"  [Rerank] Final {len(final_docs)} docs returned")
        
        return final_docs

    def _filter_docs_by_metadata(self, docs: List[Document], filter_dict: Dict) -> List[Document]:
        """Filter documents by metadata (for BM25 results which don't support native filtering)."""
        filtered = []
        for doc in docs:
            if self._matches_filter(doc.metadata, filter_dict):
                filtered.append(doc)
        return filtered
    
    def _matches_filter(self, metadata: Dict, filter_dict: Dict) -> bool:
        """Check if document metadata matches the filter."""
        if "$or" in filter_dict:
            # OR condition
            return any(self._matches_filter(metadata, sub_filter) for sub_filter in filter_dict["$or"])
        elif "$and" in filter_dict:
            # AND condition
            return all(self._matches_filter(metadata, sub_filter) for sub_filter in filter_dict["$and"])
        else:
            # Simple key-value match
            for key, value in filter_dict.items():
                if metadata.get(key) != value:
                    return False
            return True

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
    
    def with_filter(self, chroma_filter: Optional[Dict] = None, keywords: Optional[List[str]] = None) -> "FilterableHybridRetriever":
        """Return a new retriever instance with the specified filters."""
        return FilterableHybridRetriever(
            vectorstore=self.vectorstore,
            bm25_retriever=self.bm25_retriever,
            reranker=self.reranker,
            top_k_retrieval=self.top_k_retrieval,
            top_k_rerank=self.top_k_rerank,
            chroma_filter=chroma_filter,
            keyword_boost=keywords
        )


# Keep the old class for backward compatibility
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

    # 4. Return Custom Retriever (backward compatible)
    return CustomHybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        top_k_retrieval=Config.TOP_K_RETRIEVAL,
        top_k_rerank=Config.TOP_K_RERANK
    )


def get_filterable_retriever() -> FilterableHybridRetriever:
    """
    Get a filterable hybrid retriever that supports metadata filtering.
    Use .with_filter(chroma_filter, keywords) to apply filters.
    """
    # 1. Load Vector Store
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    # 2. Load BM25 Retriever
    bm25_path = os.path.join(os.path.dirname(Config.CHROMA_PERSIST_DIRECTORY), "bm25_retriever.pkl")
    if not os.path.exists(bm25_path):
        raise FileNotFoundError("BM25 retriever not found. Run ingestion first.")
    
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)

    # 3. Initialize Reranker
    reranker = HuggingFaceCrossEncoder(model_name=Config.RERANKER_MODEL_NAME)

    # 4. Return Filterable Retriever
    return FilterableHybridRetriever(
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        top_k_retrieval=Config.TOP_K_RETRIEVAL,
        top_k_rerank=Config.TOP_K_RERANK,
        chroma_filter=None,
        keyword_boost=None
    )
