import os
import uuid
import pickle
import tempfile
from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np
from pathlib import Path
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HybridRAG")

# Vector stores
import faiss
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# LangChain and document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain_community.retrievers import BM25Retriever  # Updated import

# Embedding model
from sentence_transformers import SentenceTransformer

class HybridRAG:
    """
    Advanced Hybrid RAG System with dual-retrieval (semantic and BM25) mechanism.
    
    Features:
    - Local persistence for both Qdrant and FAISS indexes
    - Optimized chunk intersection algorithm
    - Document structure-aware retrieval
    - Specialized handling for summary/overview queries
    """

    def __init__(
        self,
        data_dir: str = "./rag_data",
        collection_name: str = "document_store",
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        create_dirs: bool = True,
        use_faiss: bool = True  # Added option to disable FAISS
    ):
        """
        Initialize the RAG system with local persistence.
        
        Note on FAISS vs Qdrant:
        - Qdrant handles hybrid search combining both dense and sparse vectors
        - FAISS provides faster semantic search through optimized approximate nearest neighbor algorithms
        - We use both for their strengths, but FAISS can be disabled if Qdrant's search is sufficient
        """
        
        # Setup directory structure
        self.data_dir = Path(data_dir)
        self.qdrant_path = self.data_dir / "qdrant_db"
        self.faiss_path = self.data_dir / "faiss_index"
        self.sessions_path = self.data_dir / "sessions"
        
        if create_dirs:
            self.data_dir.mkdir(exist_ok=True)
            self.qdrant_path.mkdir(exist_ok=True)
            self.faiss_path.mkdir(exist_ok=True) 
            self.sessions_path.mkdir(exist_ok=True)
        
        # Initialize Qdrant client with local persistence
        self.client = QdrantClient(path=str(self.qdrant_path))
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Create collection if it doesn't exist
        self._initialize_collection()
        
        # Initialize FAISS index (if enabled)
        self.use_faiss = use_faiss
        if self.use_faiss:
            self.faiss_index_path = self.faiss_path / f"{collection_name}.index"
            self.faiss_mapping_path = self.faiss_path / f"{collection_name}_mapping.pkl"
            
            if self.faiss_index_path.exists() and self.faiss_mapping_path.exists():
                self._load_faiss_index()
            else:
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
                self.faiss_id_to_text_mapping = {}
                self.faiss_id_to_metadata_mapping = {}
        
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
    
    def _initialize_collection(self) -> None:
        """Initialize Qdrant collection with necessary parameters."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim, 
                    distance=Distance.COSINE
                ),
                sparse_vectors_config={
                    "text_vector_sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams()
                    )
                }
            )
            
            # Create payload indexes for efficient filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.source",
                field_schema="keyword",
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.page",
                field_schema="integer",
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.section_type",
                field_schema="keyword",
            )

    def _load_faiss_index(self) -> None:
        """Load FAISS index and mapping from disk."""
        try:
            self.faiss_index = faiss.read_index(str(self.faiss_index_path))
            
            with open(self.faiss_mapping_path, "rb") as f:
                mappings = pickle.load(f)
                self.faiss_id_to_text_mapping = mappings["text"]
                self.faiss_id_to_metadata_mapping = mappings["metadata"]
            
            logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            # Initialize empty index if loading fails
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_id_to_text_mapping = {}
            self.faiss_id_to_metadata_mapping = {}
    
    def _save_faiss_index(self) -> None:
        """Save FAISS index and mapping to disk."""
        try:
            faiss.write_index(self.faiss_index, str(self.faiss_index_path))
            
            with open(self.faiss_mapping_path, "wb") as f:
                pickle.dump({
                    "text": self.faiss_id_to_text_mapping,
                    "metadata": self.faiss_id_to_metadata_mapping
                }, f)
            
            logger.info(f"Saved FAISS index with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def _compute_sparse_embeddings(self, text: str) -> Dict[str, Any]:
        """Compute sparse BM25-style embeddings for text."""
        # Simple TF-IDF style sparse embeddings
        words = text.lower().split()
        word_set = set(words)
        
        # Compute TF (term frequency)
        tf = {}
        for word in word_set:
            tf[word] = words.count(word) / len(words)
        
        # Use hashing for indices (must be integers) and TF for values (must be floats)
        indices = [int(hash(word) % 10000) for word in word_set]  # Ensure these are integers
        values = [float(tf[word]) for word in word_set]  # Ensure these are floats
        
        # Return in the format Qdrant expects
        return {
            "indices": indices,
            "values": values
        }

    def _get_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
        """Get both dense and sparse embeddings for a list of texts."""
        dense_embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        sparse_embeddings = [self._compute_sparse_embeddings(text) for text in texts]
        
        return dense_embeddings, sparse_embeddings

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

    def ingest_documents(self, file_paths: List[str], progress_callback=None) -> str:
        """
        Ingest documents into the RAG system.
        
        This method:
        1. Loads and processes documents into intelligent chunks
        2. Generates embeddings for each chunk
        3. Stores documents in Qdrant and optionally FAISS
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
        dense_embeddings, sparse_embeddings = self._get_embeddings(texts)
        
        if progress_callback:
            progress_callback(0.4, "Adding to vector stores...")
        
        # Prepare for FAISS and BM25
        bm25_documents = []
        faiss_vectors = np.array(dense_embeddings).astype('float32')
        
        # Create Qdrant points
        points = []
        for i, (chunk, dense_emb, sparse_emb) in enumerate(zip(chunks, dense_embeddings, sparse_embeddings)):
            point_id = str(uuid.uuid4())
            
            # Create Qdrant point
            points.append(
                PointStruct(
                    id=point_id,
                    vector=dense_emb.tolist(),
                    payload={
                        "text": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                    vector_sparse=sparse_emb,
                )
            )
            
            # Add to FAISS
            faiss_id = self.faiss_index.ntotal + i
            self.faiss_id_to_text_mapping[faiss_id] = chunk.page_content
            self.faiss_id_to_metadata_mapping[faiss_id] = chunk.metadata
            
            # Add to BM25 documents
            bm25_documents.append(Document(
                page_content=chunk.page_content,
                metadata=chunk.metadata
            ))
            
            # Batch upload to Qdrant to avoid memory issues
            if len(points) >= 100:
                if progress_callback:
                    progress_callback(
                        0.4 + (0.2 * i / len(chunks)),
                        f"Adding batch to Qdrant ({i}/{len(chunks)})..."
                    )
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                points = []
        
        # Upload any remaining points to Qdrant
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        
        if progress_callback:
            progress_callback(0.6, "Updating FAISS index...")
        
        # Add vectors to FAISS
        if faiss_vectors.shape[0] > 0:
            self.faiss_index.add(faiss_vectors)
        
        # Save FAISS index
        self._save_faiss_index()
        
        if progress_callback:
            progress_callback(0.8, "Setting up BM25 retriever...")
        
        # Update BM25 retriever
        self.bm25_documents.extend(bm25_documents)
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
        Perform semantic search using either FAISS (if enabled) or Qdrant.
        
        Why both FAISS and Qdrant?
        - FAISS: Optimized for pure vector search with better performance for large collections
        - Qdrant: Provides rich filtering and combines both vector and sparse retrieval
        
        This unified interface selects the appropriate backend.
        """
        # If FAISS is enabled and initialized, use it for semantic search
        if self.use_faiss and hasattr(self, 'faiss_index') and self.faiss_index.ntotal > 0:
            # Normalize query vector for cosine similarity
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            faiss.normalize_L2(query_embedding)
            
            # Perform search
            scores, indices = self.faiss_index.search(
                query_embedding.astype(np.float32),
                k=top_k
            )
            
            # Flatten results
            scores = scores.flatten()
            indices = indices.flatten()
            
            # Convert to results format
            results = []
            for score, idx in zip(scores, indices):
                # Skip invalid indices
                if idx == -1 or idx not in self.faiss_id_to_text_mapping:
                    continue
                    
                results.append({
                    "text": self.faiss_id_to_text_mapping[idx],
                    "metadata": self.faiss_id_to_metadata_mapping[idx],
                    "score": float(score),
                    "source": "faiss"
                })
            
            return results
        else:
            # Fallback to Qdrant for semantic search if FAISS is not available
            # Create search filter if needed
            search_filter = None
            if filters:
                filter_conditions = []
                
                # Filter by source
                if "source" in filters and filters["source"]:
                    filter_conditions.append(
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=filters["source"])
                        )
                    )
                
                # Filter by page range
                if "page_start" in filters and "page_end" in filters:
                    filter_conditions.append(
                        FieldCondition(
                            key="metadata.page",
                            range=Range(
                                gte=filters["page_start"],
                                lte=filters["page_end"]
                            )
                        )
                    )
                
                # Filter by section type
                if "section_type" in filters and filters["section_type"]:
                    filter_conditions.append(
                        FieldCondition(
                            key="metadata.section_type",
                            match=MatchValue(value=filters["section_type"])
                        )
                    )
                
                if filter_conditions:
                    search_filter = Filter(
                        must=filter_conditions
                    )
            
            # Use Qdrant's vector search with dense vectors only
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                with_payload=True,
                filter=search_filter,
                search_params=models.SearchParams(
                    hnsw_ef=128,  # Higher recall at the cost of speed
                ),
            )
            
            # Extract and return results
            results = []
            for hit in search_result:
                results.append({
                    "text": hit.payload["text"],
                    "metadata": hit.payload["metadata"],
                    "score": hit.score,
                    "source": "qdrant_dense"
                })
            
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
                "metadata": doc.metadata,
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
        Perform hybrid retrieval optimizing for intersection first.
        
        Strategy:
        1. Get top-k*2 results from both semantic and BM25
        2. Prioritize results that appear in both sets
        3. Fill remaining slots with a balanced mix of semantic and BM25 results
        """
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Apply any special filters based on query type
        special_filters = self._apply_special_filters(query)
        if special_filters:
            if not filters:
                filters = special_filters
            else:
                filters.update(special_filters)
        
        # Get results from both retrievers
        semantic_results = self._semantic_search(query_embedding, top_k=top_k*2, filters=filters)
        bm25_results = self._bm25_search(query, top_k=top_k*2, filters=filters)
        
        # Create sets for checking intersection
        semantic_texts = {result["text"] for result in semantic_results}
        bm25_texts = {result["text"] for result in bm25_results}
        
        # Find intersection
        intersection_texts = semantic_texts.intersection(bm25_texts)
        
        # Prepare final results
        final_results = []
        
        # First add intersection results with boosted scores
        for result in semantic_results:
            if result["text"] in intersection_texts:
                # Find matching BM25 result to combine scores
                bm25_score = next(
                    (r["score"] for r in bm25_results if r["text"] == result["text"]),
                    0.0
                )
                
                # Boost intersection results
                combined_score = (result["score"] + bm25_score) * 1.2  # 20% boost
                
                final_results.append({
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "score": min(combined_score, 1.0),  # Cap at 1.0
                    "source": "intersection"
                })
                
                # Remove from both sets to avoid duplicates
                if result["text"] in intersection_texts:
                    intersection_texts.remove(result["text"])
        
        # Then fill remaining slots with balanced semantic and BM25 results
        remaining_slots = top_k - len(final_results)
        
        if remaining_slots > 0:
            # Get unique semantic and BM25 results (not already in final_results)
            unique_semantic_results = [
                r for r in semantic_results 
                if r["text"] not in {result["text"] for result in final_results}
            ]
            
            unique_bm25_results = [
                r for r in bm25_results 
                if r["text"] not in {result["text"] for result in final_results}
            ]
            
            # Calculate how many results to take from each source
            semantic_slots = remaining_slots // 2
            bm25_slots = remaining_slots - semantic_slots
            
            # Add balanced results
            final_results.extend(unique_semantic_results[:semantic_slots])
            final_results.extend(unique_bm25_results[:bm25_slots])
        
        # Sort by score
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        return final_results[:top_k]

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
        
        # Perform hybrid retrieval
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
        """Get list of all document sources in the collection."""
        # Using scroll to iterate through all points
        scroll_results = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            limit=100,
        )
        
        sources = set()
        while scroll_results[0]:  # While we have results
            for point in scroll_results[0]:
                if "metadata" in point.payload and "source" in point.payload["metadata"]:
                    sources.add(point.payload["metadata"]["source"])
            
            # Get next batch
            scroll_results = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=100,
                offset=scroll_results[1],  # Use offset from previous call
            )
        
        return list(sources)

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "qdrant_points": 0,
            "faiss_vectors": self.faiss_index.ntotal if hasattr(self, "faiss_index") else 0,
            "bm25_documents": len(self.bm25_documents) if hasattr(self, "bm25_documents") else 0,
            "sources": self.get_document_sources(),
            "sessions": len(self.sessions),
        }
        
        # Get Qdrant points count
        try:
            collection_info = self.client.get_collection(self.collection_name)
            stats["qdrant_points"] = collection_info.vectors_count
        except Exception as e:
            logger.error(f"Failed to get Qdrant collection info: {e}")
        
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
