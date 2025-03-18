def reciprocal_rank_fusion(results_lists, k=60):
    """
    Implements reciprocal rank fusion to combine multiple result lists.
    
    Args:
        results_lists: List of lists of Document objects
        k: Constant to prevent division by zero for top results
        
    Returns:
        List of Document objects sorted by RRF score
    """
    # Track document scores by content hash
    doc_scores = defaultdict(float)
    
    # Process results from each retriever
    for results in results_lists:
        # Track the rank of each document
        for rank, doc in enumerate(results):
            # Use content hash as ID to detect duplicates
            doc_id = hash(doc.page_content)
            
            # RRF formula: 1 / (rank + k)
            doc_scores[doc_id] += 1.0 / (rank + k)
    
    # Create mapping from ID to document
    id_to_doc = {}
    retrieved_by = defaultdict(list)
    
    for i, results in enumerate(results_lists):
        for doc in results:
            doc_id = hash(doc.page_content)
            
            # Track which retrievers found this document
            retriever_name = doc.metadata.get("retriever", f"retriever_{i}")
            if retriever_name not in retrieved_by[doc_id]:
                retrieved_by[doc_id].append(retriever_name)
            
            # Keep the document with the highest score if duplicated
            if doc_id not in id_to_doc:
                id_to_doc[doc_id] = doc
            elif "score" in doc.metadata and "score" in id_to_doc[doc_id].metadata:
                if doc.metadata["score"] > id_to_doc[doc_id].metadata["score"]:
                    id_to_doc[doc_id] = doc
    
    # Add RRF score and retriever info to metadata
    for doc_id in id_to_doc:
        id_to_doc[doc_id].metadata["rrf_score"] = doc_scores[doc_id]
        id_to_doc[doc_id].metadata["retrieved_by"] = "+".join(retrieved_by[doc_id])
        
        # If a document is found by multiple retrievers, boost its score
        if len(retrieved_by[doc_id]) > 1:
            id_to_doc[doc_id].metadata["fusion_boost"] = True
            
    # Sort by RRF score and return documents
    sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    return [id_to_doc[doc_id] for doc_id in sorted_ids]class FAISSRetriever(BaseRetriever):
    """Custom FAISS retriever implementation for Langchain compatibility."""
    
    def __init__(
        self, 
        faiss_index, 
        documents, 
        id_to_embedding,
        embedding_model, 
        k=5,
        filters=None
    ):
        """
        Initialize the FAISS retriever.
        
        Args:
            faiss_index: FAISS index object
            documents: Dictionary of document id -> {"text": str, "metadata": dict}
            id_to_embedding: Dictionary of document id -> embedding vector
            embedding_model: Sentence Transformer model for encoding queries
            k: Number of documents to retrieve
            filters: Optional filters to apply to results
        """
        super().__init__()
        self.faiss_index = faiss_index
        self.documents = documents
        self.id_to_embedding = id_to_embedding
        self.embedding_model = embedding_model
        self.k = k
        self.filters = filters
        
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents relevant to the query."""
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Normalize for cosine similarity
        query_embedding_reshaped = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding_reshaped)
        query_embedding = query_embedding_reshaped[0]
        
        # Search FAISS
        k = min(self.k*2, self.faiss_index.ntotal)  # Don't request more than we have
        if k == 0:
            return []
            
        scores, indices = self.faiss_index.search(
            query_embedding.astype(np.float32).reshape(1, -1),
            k=k
        )
        
        # Flatten results
        scores = scores.flatten()
        indices = indices.flatten()
        
        # Convert to Langchain documents
        results = []
        doc_ids_seen = set()
        
        for score, idx in zip(scores, indices):
            if idx == -1:  # Invalid index
                continue
                
            # Find document by matching text
            for doc_id, doc in self.documents.items():
                if doc_id in doc_ids_seen:
                    continue
                    
                # Apply filters if needed
                if self.filters:
                    skip = False
                    
                    # Filter by source
                    if "source" in self.filters and self.filters["source"]:
                        if doc["metadata"].get("source") != self.filters["source"]:
                            skip = True
                    
                    # Filter by page range
                    if "page_start" in self.filters and "page_end" in self.filters:
                        page = doc["metadata"].get("page", 0)
                        if page < self.filters["page_start"] or page > self.filters["page_end"]:
                            skip = True
                    
                    # Filter by section type
                    if "section_type" in self.filters and self.filters["section_type"]:
                        if doc["metadata"].get("section_type") != self.filters["section_type"]:
                            skip = True
                    
                    if skip:
                        continue
                
                # Match found, create document
                langchain_doc = Document(
                    page_content=doc["text"],
                    metadata={
                        **doc["metadata"],
                        "score": float(score),
                        "retriever": "faiss"
                    }
                )
                results.append(langchain_doc)
                doc_ids_seen.add(doc_id)
                
                # Break after finding match
                break
            
            # Stop if we have enough results
            if len(results) >= self.k:
                break
                
        return resultsimport os
import uuid
import pickle
import tempfile
from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np
from pathlib import Path
import time
import logging
import json
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HybridRAG")

# Vector stores
import faiss

# LangChain and document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.merger_retriever import MergerRetriever

# Embedding model
from sentence_transformers import SentenceTransformer

class HybridRAG:
    """
    Advanced Hybrid RAG System with fusion retrieval mechanism.
    Uses FAISS for vector storage and pickle for document storage.
    
    Features:
    - Local persistence for FAISS indexes and document store
    - Reciprocal Rank Fusion for combining multiple retrievers
    - Document structure-aware retrieval
    - Specialized handling for summary/overview queries
    """

    def __init__(
        self,
        data_dir: str = "./rag_data",
        collection_name: str = "document_store",
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        create_dirs: bool = True
    ):
        """Initialize the RAG system with local persistence."""
        
        # Setup directory structure
        self.data_dir = Path(data_dir)
        self.faiss_path = self.data_dir / "faiss_index"
        self.docs_path = self.data_dir / "documents"
        self.sessions_path = self.data_dir / "sessions"
        
        if create_dirs:
            self.data_dir.mkdir(exist_ok=True)
            self.faiss_path.mkdir(exist_ok=True)
            self.docs_path.mkdir(exist_ok=True)
            self.sessions_path.mkdir(exist_ok=True)
        
        # Store collection name for file naming
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.faiss_index_path = self.faiss_path / f"{collection_name}.index"
        self.documents_store_path = self.docs_path / f"{collection_name}_docs.pkl"
        
        # Load existing index and documents or create new ones
        if self.faiss_index_path.exists() and self.documents_store_path.exists():
            self._load_faiss_index()
            self._load_document_store()
        else:
            # Initialize FAISS index for cosine similarity
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Initialize document store
            # Structure: {id -> {"text": text, "metadata": metadata}}
            self.documents = {}
            self.id_to_embedding = {}  # Store embeddings for retrieval scoring
        
        # Initialize BM25 retriever
        self.bm25_documents = []
        self.bm25_retriever = None
        
        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        # Session management
        self.sessions = {}
        self._load_sessions()
        
        # Document source tracking
        self.sources = set()
        self._load_sources()
    
    def _save_faiss_index(self) -> None:
        """Save FAISS index to disk."""
        try:
            faiss.write_index(self.faiss_index, str(self.faiss_index_path))
            logger.info(f"Saved FAISS index with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def _load_faiss_index(self) -> None:
        """Load FAISS index from disk."""
        try:
            self.faiss_index = faiss.read_index(str(self.faiss_index_path))
            logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            # Initialize empty index if loading fails
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
    
    def _save_document_store(self) -> None:
        """Save document store to disk."""
        try:
            with open(self.documents_store_path, "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "id_to_embedding": self.id_to_embedding
                }, f)
            logger.info(f"Saved document store with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Failed to save document store: {e}")
    
    def _load_document_store(self) -> None:
        """Load document store from disk."""
        try:
            with open(self.documents_store_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data.get("documents", {})
                self.id_to_embedding = data.get("id_to_embedding", {})
            logger.info(f"Loaded document store with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Failed to load document store: {e}")
            self.documents = {}
            self.id_to_embedding = {}
    
    def _save_sources(self) -> None:
        """Save document sources to disk."""
        try:
            sources_path = self.docs_path / f"{self.collection_name}_sources.pkl"
            with open(sources_path, "wb") as f:
                pickle.dump(list(self.sources), f)
        except Exception as e:
            logger.error(f"Failed to save sources: {e}")
    
    def _load_sources(self) -> None:
        """Load document sources from disk."""
        try:
            sources_path = self.docs_path / f"{self.collection_name}_sources.pkl"
            if sources_path.exists():
                with open(sources_path, "rb") as f:
                    self.sources = set(pickle.load(f))
            else:
                self.sources = set()
                
            # If sources is empty but we have documents, rebuild the sources
            if not self.sources and hasattr(self, 'documents') and self.documents:
                for doc_id, doc_data in self.documents.items():
                    source = doc_data["metadata"].get("source", "")
                    if source:
                        self.sources.add(source)
                self._save_sources()
                
        except Exception as e:
            logger.error(f"Failed to load sources: {e}")
            self.sources = set()

    def _classify_document_section(self, text: str, page_num: int, doc_source: str) -> str:
        """Classify document section based on content and position."""
        text_lower = text.lower()
        
        # Check for table of contents or index
        if any(marker in text_lower for marker in ["contents", "table of contents", "index", "appendix"]):
            return "toc"
        
        # Check for introduction or abstract
        if any(marker in text_lower for marker in ["introduction", "abstract", "overview", "summary", "preface"]):
            return "introduction"
        
        # Check for conclusion
        if any(marker in text_lower for marker in ["conclusion", "summary", "final", "results", "discussion"]):
            return "conclusion"
        
        # Check for first few pages (likely introduction)
        if page_num <= 2:
            return "introduction"
        
        # Default to content
        return "content"

    def _create_document_chunks(self, documents: List[Document]) -> List[Document]:
        """Create intelligent document chunks with metadata."""
        all_chunks = []
        
        # Group documents by source
        doc_groups = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in doc_groups:
                doc_groups[source] = []
            doc_groups[source].append(doc)
        
        # Process each document group separately
        for source, docs in doc_groups.items():
            # Sort by page number
            docs.sort(key=lambda x: x.metadata.get("page", 0))
            
            # Process each page
            for doc in docs:
                page_num = doc.metadata.get("page", 0)
                
                # Special chunking for early pages and TOC/index
                if page_num <= 2:
                    # Use smaller chunks for better granularity on important pages
                    intro_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=200,
                        chunk_overlap=50,
                        separators=["\n\n", "\n", ". ", " ", ""],
                        length_function=len,
                    )
                    page_chunks = intro_splitter.split_documents([doc])
                else:
                    # Regular chunking for other pages
                    page_chunks = self.text_splitter.split_documents([doc])
                
                # Classify each chunk and enhance metadata
                for chunk in page_chunks:
                    section_type = self._classify_document_section(
                        chunk.page_content, page_num, source
                    )
                    
                    # Update metadata
                    chunk.metadata["section_type"] = section_type
                    
                    # Add to chunks list
                    all_chunks.append(chunk)
        
        # Sort all chunks by source and page
        all_chunks.sort(key=lambda x: (
            x.metadata.get("source", ""),
            x.metadata.get("page", 0)
        ))
        
        return all_chunks
    
    def _compute_bm25_scores(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute BM25 scores for each document and add as a score field."""
        # Create a mini BM25 retriever just for these documents
        bm25_docs = [Document(page_content=doc["text"], metadata=doc["metadata"]) for doc in documents]
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        
        # Get scores
        scored_docs = bm25_retriever.get_relevant_documents(query)
        
        # Map back to original documents with scores
        results = []
        for i, doc in enumerate(scored_docs):
            # Normalized score between 0 and 1 based on position
            score = 1.0 - (i / len(scored_docs) if len(scored_docs) > 1 else 0)
            
            # Find matching original document
            for original_doc in documents:
                if original_doc["text"] == doc.page_content:
                    results.append({
                        "text": original_doc["text"],
                        "metadata": original_doc["metadata"],
                        "score": score,
                        "source": "bm25"
                    })
                    break
        
        return results

    def ingest_documents(self, file_paths: List[str], progress_callback=None) -> str:
        """
        Ingest documents into the RAG system.
        
        This method:
        1. Loads and processes documents into intelligent chunks
        2. Generates embeddings for each chunk
        3. Stores documents in FAISS and document store
        4. Sets up BM25 for keyword retrieval
        
        Args:
            file_paths: List of file paths to ingest
            progress_callback: Optional callback function for progress updates
                               Signature: callback(progress_percent, status_message)
        """
        all_docs = []
        
        # Load documents
        for i, file_path in enumerate(file_paths):
            try:
                if progress_callback:
                    progress_callback(
                        (i / len(file_paths)) * 0.2,  # First 20% for loading
                        f"Loading {os.path.basename(file_path)}..."
                    )
                
                # Handle PDF documents
                if file_path.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    
                    # Add source metadata
                    for doc in docs:
                        doc.metadata["source"] = os.path.basename(file_path)
                    
                    all_docs.extend(docs)
                
                # Handle text documents
                elif file_path.lower().endswith(".txt"):
                    loader = TextLoader(file_path)
                    docs = loader.load()
                    
                    # Add source metadata
                    for doc in docs:
                        doc.metadata["source"] = os.path.basename(file_path)
                        doc.metadata["page"] = 0  # Default page for text files
                    
                    all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                if progress_callback:
                    progress_callback(
                        (i / len(file_paths)) * 0.2,
                        f"Error loading {os.path.basename(file_path)}: {str(e)}"
                    )
        
        if progress_callback:
            progress_callback(0.2, "Creating intelligent chunks...")
        
        # Create intelligent chunks with enhanced metadata
        chunks = self._create_document_chunks(all_docs)
        
        if progress_callback:
            progress_callback(0.3, "Generating embeddings...")
        
        # Get embeddings for all chunks
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        
        if progress_callback:
            progress_callback(0.4, "Adding to vector store...")
        
        # Prepare for FAISS
        faiss_vectors = np.array(embeddings).astype('float32')
        
        # Create document entries and track sources
        bm25_documents = []
        current_index = self.faiss_index.ntotal
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create unique ID
            doc_id = str(uuid.uuid4())
            
            # Store in document store
            self.documents[doc_id] = {
                "text": chunk.page_content,
                "metadata": chunk.metadata
            }
            
            # Store embedding for scoring
            self.id_to_embedding[doc_id] = embedding.tolist()
            
            # Track FAISS index mapping
            faiss_id = current_index + i
            
            # Add to BM25 documents
            bm25_documents.append(Document(
                page_content=chunk.page_content,
                metadata={"doc_id": doc_id, **chunk.metadata}
            ))
            
            # Track document source
            source = chunk.metadata.get("source", "")
            if source:
                self.sources.add(source)
            
            # Update progress periodically
            if progress_callback and i % 20 == 0:
                progress_callback(
                    0.4 + (0.3 * i / len(chunks)),
                    f"Processing documents ({i+1}/{len(chunks)})..."
                )
        
        if progress_callback:
            progress_callback(0.7, "Updating FAISS index...")
        
        # Add vectors to FAISS
        if faiss_vectors.shape[0] > 0:
            self.faiss_index.add(faiss_vectors)
        
        # Save FAISS index
        self._save_faiss_index()
        
        # Save document store
        self._save_document_store()
        
        # Save sources
        self._save_sources()
        
        if progress_callback:
            progress_callback(0.8, "Setting up BM25 retriever...")
        
        # Update BM25 retriever
        self.bm25_documents = bm25_documents
        self.bm25_retriever = BM25Retriever.from_documents(
            self.bm25_documents,
            preprocess_func=lambda text: text.lower()  # Case-insensitive search
        )
        
        if progress_callback:
            progress_callback(1.0, "Ingestion complete!")
        
        return f"Successfully ingested {len(chunks)} chunks from {len(file_paths)} documents."

    def _semantic_search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using FAISS.
        """
        # Normalize query vector for cosine similarity
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        faiss.normalize_L2(query_embedding)
        
        # Perform search
        scores, indices = self.faiss_index.search(
            query_embedding.astype(np.float32),
            k=min(top_k*2, self.faiss_index.ntotal)  # Don't request more than we have
        )
        
        # Flatten results
        scores = scores.flatten()
        indices = indices.flatten()
        
        # Convert to results format
        results = []
        for score, idx in zip(scores, indices):
            # Skip invalid indices
            if idx == -1:
                continue
            
            # Find document by index
            doc_id = None
            for d_id, emb in self.id_to_embedding.items():
                if self.documents[d_id]["text"] == self.bm25_documents[idx].page_content:
                    doc_id = d_id
                    break
            
            if not doc_id:
                continue
                
            doc = self.documents.get(doc_id)
            if not doc:
                continue
            
            # Apply filters if needed
            if filters:
                skip = False
                
                # Filter by source
                if "source" in filters and filters["source"] and doc["metadata"].get("source") != filters["source"]:
                    skip = True
                
                # Filter by page range
                if "page_start" in filters and "page_end" in filters:
                    page = doc["metadata"].get("page", 0)
                    if page < filters["page_start"] or page > filters["page_end"]:
                        skip = True
                
                # Filter by section type
                if "section_type" in filters and filters["section_type"] and doc["metadata"].get("section_type") != filters["section_type"]:
                    skip = True
                
                if skip:
                    continue
                    
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(score),
                "source": "faiss"
            })
            
            # Break if we have enough results after filtering
            if len(results) >= top_k:
                break
        
        return results

    def _bm25_search(
        self, 
        query: str, 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform keyword search using BM25."""
        if not self.bm25_retriever:
            return []
        
        # Get BM25 results
        bm25_results = self.bm25_retriever.get_relevant_documents(query, k=top_k * 2)  # Get more for filtering
        
        # Apply filters if needed
        if filters:
            filtered_results = []
            for doc in bm25_results:
                matches_filters = True
                
                # Filter by source
                if "source" in filters and filters["source"]:
                    if doc.metadata.get("source") != filters["source"]:
                        matches_filters = False
                
                # Filter by page range
                if "page_start" in filters and "page_end" in filters:
                    page = doc.metadata.get("page", 0)
                    if page < filters["page_start"] or page > filters["page_end"]:
                        matches_filters = False
                
                # Filter by section type
                if "section_type" in filters and filters["section_type"]:
                    if doc.metadata.get("section_type") != filters["section_type"]:
                        matches_filters = False
                
                if matches_filters:
                    filtered_results.append(doc)
            
            bm25_results = filtered_results[:top_k]
        else:
            bm25_results = bm25_results[:top_k]
        
        # Convert to results format
        results = []
        for i, doc in enumerate(bm25_results):
            # Assign score based on rank (normalized to 0-1)
            score = 1.0 - (i / len(bm25_results) if len(bm25_results) > 1 else 0)
            
            results.append({
                "text": doc.page_content,
                "metadata": {k: v for k, v in doc.metadata.items() if k != "doc_id"},
                "score": score,
                "source": "bm25"
            })
        
        return results

    def _apply_special_filters(self, query: str) -> Optional[Dict[str, Any]]:
        """Apply special filters based on query type."""
        query_lower = query.lower()
        
        # Detect "what is this document about" type queries
        if any(phrase in query_lower for phrase in [
            "what is", "what's this", "about", "summary", "summarize",
            "overview", "describe", "introduction", "tell me about"
        ]):
            # For summary queries, prioritize introduction and early pages
            return {
                "section_type": "introduction"
            }
        
        # Detect conclusion queries
        if any(phrase in query_lower for phrase in [
            "conclusion", "findings", "results", "outcome", "end result",
            "final", "summary of findings"
        ]):
            return {
                "section_type": "conclusion"
            }
            
        # Detect TOC/index queries
        if any(phrase in query_lower for phrase in [
            "table of contents", "index", "chapters", "sections", "structure",
            "layout", "organization"
        ]):
            return {
                "section_type": "toc"
            }
        
        # No special filters for regular queries
        return None

    def _hybrid_retrieval(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval using LangChain's retriever fusion.
        
        This implementation uses Reciprocal Rank Fusion to combine results
        from multiple retrievers in a theoretically sound way.
        """
        # Apply any special filters based on query type
        special_filters = self._apply_special_filters(query)
        if special_filters:
            if not filters:
                filters = special_filters
            else:
                filters.update(special_filters)
                
        # Create FAISS retriever
        faiss_retriever = FAISSRetriever(
            faiss_index=self.faiss_index,
            documents=self.documents,
            id_to_embedding=self.id_to_embedding,
            embedding_model=self.embedding_model,
            k=top_k,
            filters=filters
        )
        
        # Use existing BM25 retriever or create one
        if not self.bm25_retriever and self.bm25_documents:
            self.bm25_retriever = BM25Retriever.from_documents(
                self.bm25_documents,
                preprocess_func=lambda text: text.lower()
            )
        
        # If BM25 retriever is not available, use only FAISS
        if not self.bm25_retriever:
            docs = faiss_retriever.get_relevant_documents(query)
            
            # Convert to result format
            results = []
            for doc in docs:
                results.append({
                    "text": doc.page_content,
                    "metadata": {k: v for k, v in doc.metadata.items() if k not in ["score", "retriever"]},
                    "score": doc.metadata.get("score", 0.5),
                    "source": doc.metadata.get("retriever", "faiss")
                })
            return results
            
        # Set up retriever with appropriate weights
        bm25_docs = self.bm25_retriever.get_relevant_documents(query, k=top_k)
        
        # Add retriever info to metadata
        for doc in bm25_docs:
            doc.metadata["retriever"] = "bm25"
            
            # Add score based on position (normalize to 0-1)
            for i, d in enumerate(bm25_docs):
                d.metadata["score"] = 1.0 - (i / len(bm25_docs) if len(bm25_docs) > 1 else 0)
        
        # Get results from both retrievers
        faiss_docs = faiss_retriever.get_relevant_documents(query)
        
        # Apply Reciprocal Rank Fusion
        fused_docs = reciprocal_rank_fusion([faiss_docs, bm25_docs])
        
        # Limit to top_k
        fused_docs = fused_docs[:top_k]
        
        # Convert to result format
        results = []
        for doc in fused_docs:
            source = "fusion"
            if "retrieved_by" in doc.metadata:
                source = doc.metadata["retrieved_by"]
                
            # Determine score (prefer RRF score if available)
            if "rrf_score" in doc.metadata:
                score = doc.metadata["rrf_score"]
            elif "score" in doc.metadata:
                score = doc.metadata["score"]
            else:
                score = 0.5
                
            # Apply boost for documents found by multiple retrievers
            if doc.metadata.get("fusion_boost", False):
                score *= 1.2  # 20% boost for intersection
            
            # Create result entry
            results.append({
                "text": doc.page_content,
                "metadata": {k: v for k, v in doc.metadata.items() 
                             if k not in ["score", "retriever", "rrf_score", "retrieved_by", "fusion_boost"]},
                "score": score,
                "source": source
            })
        
        return results

    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank search results based on query relevance and document structure."""
        query_lower = query.lower()
        
        # Score key document sections higher based on query type
        for result in results:
            section_type = result["metadata"].get("section_type", "content")
            page_num = result["metadata"].get("page", 0)
            
            # Boost introductions for "what's this about" queries
            if any(phrase in query_lower for phrase in ["what is", "what's this", "about", "summary"]):
                if section_type == "introduction":
                    result["score"] *= 1.3  # 30% boost
                elif section_type == "toc":
                    result["score"] *= 1.2  # 20% boost
                elif page_num <= 2:
                    result["score"] *= 1.1  # 10% boost
            
            # Boost conclusions for conclusion-related queries
            elif any(phrase in query_lower for phrase in ["conclusion", "findings", "results"]):
                if section_type == "conclusion":
                    result["score"] *= 1.3  # 30% boost
            
            # Boost TOC for structure-related queries
            elif any(phrase in query_lower for phrase in ["contents", "chapters", "sections"]):
                if section_type == "toc":
                    result["score"] *= 1.3  # 30% boost
        
        # Re-sort based on adjusted scores
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results

    def _generate_context(self, results: List[Dict[str, Any]], max_tokens: int = 3000) -> str:
        """Generate context string from search results, limiting to max tokens."""
        context_parts = []
        current_length = 0
        
        # Sort results by page number for coherent context
        sorted_results = sorted(
            results, 
            key=lambda x: (
                x["metadata"].get("source", ""), 
                x["metadata"].get("page", 0)
            )
        )
        
        for result in sorted_results:
            # Rough token estimation (words / 0.75)
            result_tokens = len(result["text"].split()) // 0.75
            
            if current_length + result_tokens <= max_tokens:
                source = result["metadata"].get("source", "Unknown")
                page = result["metadata"].get("page", 0)
                section = result["metadata"].get("section_type", "content")
                retrieval_method = result.get("source", "hybrid")
                
                # Format the context with source information
                context_part = (
                    f"\n--- From {source} (Page {page}, {section.capitalize()}) "
                    f"[Score: {result['score']:.2f}, Method: {retrieval_method}] ---\n"
                    f"{result['text']}\n"
                )
                context_parts.append(context_part)
                current_length += result_tokens
            else:
                break
        
        return "\n".join(context_parts)

    def generate_answer(
        self, 
        query: str, 
        session_id: str = None,
        abc_response_func: callable = None,
        top_k: int = 5,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate an answer for the given query using the external LLM function."""
        
        # Create session if it doesn't exist
        if session_id and session_id not in self.sessions:
            self.sessions[session_id] = {"history": []}
        
        # Get chat history for context if available
        chat_history = ""
        if session_id and self.sessions[session_id]["history"]:
            # Format last 3 exchanges for context
            history = self.sessions[session_id]["history"][-3:]
            chat_history = "\n".join([f"Human: {h['question']}\nAssistant: {h['answer']}" for h in history])
            chat_history += "\n"
        
        # Perform hybrid retrieval using Langchain Fusion
        search_results = self._hybrid_retrieval(
            query=query,
            top_k=top_k,
        )
        
        # Rerank results based on query and document structure
        reranked_results = self._rerank_results(query, search_results)
        
        # Generate context from search results
        context = self._generate_context(reranked_results)
        
        # Construct the prompt template
        prompt_template = """You are a helpful assistant that answers questions based on the provided context.

Chat History:
{chat_history}

Context:
{context}

Human Question: {question}

Answer the question based on the context provided. If the answer is not in the context, say "I don't have enough information to answer this question." Ensure your answer is accurate, comprehensive, and directly addresses the question.

Answer:"""

        # Create the prompt with context and query
        prompt = prompt_template.format(
            chat_history=chat_history,
            context=context,
            question=query,
        )
        
        # Use the external LLM function to generate the answer
        if abc_response_func:
            answer = abc_response_func(prompt)
        else:
            # Fallback if no LLM function is provided
            answer = "Error: LLM function not provided. Please ensure you've imported and passed your abc_response function."
        
        # Update session history
        if session_id:
            self.sessions[session_id]["history"].append({
                "question": query,
                "answer": answer,
            })
            
            # Save session after update
            self._save_sessions()
        
        # Return the answer and the search results for citation in the UI
        return answer, reranked_results

    def get_document_sources(self) -> List[str]:
        """Get list of all document sources."""
        return list(self.sources)

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "faiss_vectors": self.faiss_index.ntotal if hasattr(self, 'faiss_index') else 0,
            "document_count": len(self.documents) if hasattr(self, 'documents') else 0,
            "bm25_documents": len(self.bm25_documents) if hasattr(self, 'bm25_documents') else 0,
            "sources": self.get_document_sources(),
            "sessions": len(self.sessions),
        }
        
        return stats

    def clear_session(self, session_id: str) -> None:
        """Clear a specific session history."""
        if session_id in self.sessions:
            self.sessions[session_id] = {"history": []}
            self._save_sessions()
    
    def _save_sessions(self) -> None:
        """Save all sessions to disk."""
        try:
            session_file = self.sessions_path / "sessions.pkl"
            with open(session_file, "wb") as f:
                pickle.dump(self.sessions, f)
            logger.info(f"Sessions saved to {session_file}")
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
    
    def _load_sessions(self) -> None:
        """Load sessions from disk."""
        session_file = self.sessions_path / "sessions.pkl"
        if session_file.exists():
            try:
                with open(session_file, "rb") as f:
                    self.sessions = pickle.load(f)
                logger.info(f"Loaded {len(self.sessions)} sessions from {session_file}")
            except Exception as e:
                logger.error(f"Failed to load sessions: {e}")
                self.sessions = {}
        else:
            self.sessions = {}
