import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, MultiVectorComparator, MultiVectorConfig, SparseVectorParams
import pickle
import hashlib
import os
import re
import uuid
from pathlib import Path
import io
import time

# Import FastEmbed for hybrid search
try:
    from fastembed import TextEmbedding, SparseTextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    st.warning("FastEmbed package not available. Hybrid search will use fallback method.")
    st.info("To enable full hybrid search, install fastembed: pip install fastembed")

# Use Qdrant as the vector database
USE_FAISS = False  # Set to False to use Qdrant

# Create directories
VECTORDB_DIR = Path("./vectordb")
VECTORDB_DIR.mkdir(exist_ok=True, parents=True)
LOCAL_MODEL_PATH = "./models/all-MiniLM-L6-v2"
os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

# Page configuration
st.set_page_config(page_title="RAG PDF Chat", layout="wide")
st.title("Chat with PDF using Local LLM")

# Sidebar for PDF upload and settings
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
    
    # Display which implementation is being used
    if USE_FAISS:
        st.info("Currently using: FAISS")
    else:
        if FASTEMBED_AVAILABLE:
            st.success("Using: Qdrant with BM25 + Vector Hybrid Search")
        else:
            st.info("Using: Qdrant with Text + Vector Search")
            st.warning("Install fastembed package for better hybrid search support")
        
        # Add Qdrant retrieval settings
        st.subheader("Qdrant Search Settings")
        
        retrieval_method = st.radio(
            "Retrieval Method",
            options=["Ensemble (Best Performance)", "Vector Only", "Hybrid Only"],
            index=0,
            help="Select the retrieval method to use",
            key="retrieval_method_radio"
        )
        
        st.session_state.use_ensemble = (retrieval_method == "Ensemble (Best Performance)")
        st.session_state.use_vector_only = (retrieval_method == "Vector Only")
        st.session_state.use_hybrid_only = (retrieval_method == "Hybrid Only")
        
        st.session_state.hybrid_weight = st.slider(
            "Semantic Search Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Balance between keyword search (0.0) and semantic search (1.0)",
            key="semantic_weight_slider"
        )
        
        st.session_state.result_count = st.slider(
            "Number of results",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of document chunks to retrieve for answering",
            key="result_count_slider"
        )
        
        st.session_state.use_reranking = st.checkbox(
            "Use Reranking",
            value=True,
            help="Rerank results to improve relevance and diversity",
            key="reranking_checkbox"
        )
    
    # Clear chat button
    if st.button("Clear Chat", key="clear_chat_button"):
        st.session_state.messages = []
        
        if USE_FAISS:
            st.session_state.retriever = None
        else:
            st.session_state.collection_name = None
            
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if USE_FAISS and "retriever" not in st.session_state:
    st.session_state.retriever = None
    
if not USE_FAISS and "collection_name" not in st.session_state:
    st.session_state.collection_name = None

# Text cleaning utility function (from BestRAG implementation)
def clean_text(text: str) -> str:
    """
    Clean the input text by removing special characters and formatting.
    """
    text = re.sub(r'_+', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'(\d+\.)\s', r'\n\1 ', text)
    text = re.sub(r'[●■○]', '', text)
    text = re.sub(r'[""''«»]', '"', text)
    text = re.sub(r'[–—−]', '-', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Initialize models for embeddings
@st.cache_resource
def load_sentence_transformer():
    """Initialize the embedding model from local path or download if not present."""
    try:
        with st.spinner("Loading embedding model..."):
            model = SentenceTransformer(LOCAL_MODEL_PATH)
            st.success("✅ Model loaded from local path")
    except Exception as e:
        with st.spinner(f"Model not found locally or error loading. Downloading model (this may take a moment)..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            # Save the model for future use
            os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
            model.save(LOCAL_MODEL_PATH)
            st.success("✅ Model downloaded and saved locally")
    return model

# Initialize dense embedding model with FastEmbed
@st.cache_resource
def load_dense_model():
    """Initialize the dense embedding model if FastEmbed is available."""
    if not FASTEMBED_AVAILABLE:
        return None
    
    try:
        with st.spinner("Loading dense embedding model..."):
            # Using fastembed's TextEmbedding for dense vectors
            model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
            st.success("✅ Dense model loaded")
            return model
    except Exception as e:
        st.warning(f"Error loading dense model: {e}")
        return None

# Initialize BM25 embedding model if FastEmbed is available
@st.cache_resource
def load_bm25_model():
    """Initialize the BM25 embedding model if FastEmbed is available."""
    if not FASTEMBED_AVAILABLE:
        return None
    
    try:
        with st.spinner("Loading BM25 embedding model..."):
            model = SparseTextEmbedding("Qdrant/bm25")
            st.success("✅ BM25 model loaded")
            return model
    except Exception as e:
        st.warning(f"Error loading BM25 model: {e}")
        return None

# Setup Qdrant client
@st.cache_resource
def setup_qdrant_client():
    """Setup Qdrant client with local persistence."""
    try:
        client = QdrantClient(path=str(VECTORDB_DIR / "qdrant_db"))
        return client
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {e}")
        return None

# Create collection for the PDF if it doesn't exist (Qdrant)
def create_collection(client, collection_name, vector_size=384):
    """Create a new collection if it doesn't exist with advanced hybrid search capabilities."""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            # Create collection with optimized configuration for hybrid search based on BestRAG approach
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense-vector": models.VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    ),
                    "output-token-embeddings": models.VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                        multivector_config=MultiVectorConfig(
                            comparator=MultiVectorComparator.MAX_SIM
                        )
                    ),
                },
                sparse_vectors_config={"sparse": SparseVectorParams()},
                # Optimizers for better performance
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=20000,
                    indexing_threshold=20000
                )
            )
            st.success(f"✅ Collection '{collection_name}' created with advanced hybrid search capabilities")
        else:
            st.info(f"Using existing collection '{collection_name}'")
            
        # Configure payload indexing for text and metadata search
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="content",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True
                )
            )
        except Exception:
            # Index may already exist, continue
            pass
        
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="page_num",
                field_schema=models.PayloadSchemaType.INTEGER
            )
        except Exception:
            # Index may already exist, continue
            pass
        
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="source_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        except Exception:
            # Index may already exist, continue
            pass
            
    except Exception as e:
        st.error(f"Error creating collection: {e}")
        raise e

# Process PDF and add to Qdrant with hybrid search support (based on BestRAG implementation)
def process_pdf_qdrant(file_bytes, collection_name, progress_container=None):
    try:
        # Extract text
        reader = PdfReader(io.BytesIO(file_bytes))
        all_chunks = []
        
        # Initialize FastEmbed models if available
        dense_model = load_dense_model() if FASTEMBED_AVAILABLE else None
        bm25_model = load_bm25_model() if FASTEMBED_AVAILABLE else None
        
        # Process each page with page number tracking
        if progress_container:
            with progress_container:
                st.info(f"Processing PDF with {len(reader.pages)} pages...")
                
                # Extract pages
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():  # Only process non-empty pages
                        # Clean the text
                        clean_page_text = clean_text(page_text)
                        
                        # Add page metadata
                        all_chunks.append({
                            "content": clean_page_text,
                            "page_num": page_num + 1,
                            "is_page_start": True,
                            "is_page_end": True,
                            "source_type": "page"
                        })
                
                # Split text for more granular chunks
                st.info("Creating smaller chunks for more precise retrieval...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                
                detailed_chunks = []
                for page_chunk in all_chunks:
                    smaller_chunks = text_splitter.split_text(page_chunk["content"])
                    for idx, chunk in enumerate(smaller_chunks):
                        detailed_chunks.append({
                            "content": chunk,
                            "chunk_id": f"p{page_chunk['page_num']}_c{idx}",
                            "page_num": page_chunk["page_num"],
                            "position": idx,
                            "is_page_start": idx == 0,
                            "is_page_end": idx == len(smaller_chunks) - 1,
                            "source_type": "chunk"
                        })
                
                # Combine with page-level chunks for multi-granularity search
                all_chunks.extend(detailed_chunks)
                
                # Get Qdrant client
                qdrant_client = setup_qdrant_client()
                
                # Check if collection exists and has points
                collection_info = qdrant_client.get_collection(collection_name)
                existing_count = collection_info.points_count
                
                # Skip if chunks are already added
                if existing_count > 0:
                    st.info(f"Document chunks already added to collection (found {existing_count} points)")
                    return
                
                # Generate embeddings and upload to Qdrant
                st.info(f"Generating embeddings for {len(all_chunks)} chunks...")
                
                if FASTEMBED_AVAILABLE and dense_model and bm25_model:
                    # Using FastEmbed for hybrid search with dense and BM25 embeddings
                    st.info("Using FastEmbed for hybrid search with dense and BM25 embeddings")
                    
                    # Extract content from chunks
                    contents = [chunk["content"] for chunk in all_chunks]
                    
                    # Generate embeddings with progress tracking
                    st.info("Generating dense embeddings...")
                    dense_progress = st.progress(0)
                    
                    # Process embeddings in batches
                    batch_size = 20
                    all_dense_embeddings = []
                    
                    for i in range(0, len(contents), batch_size):
                        batch = contents[i:min(i+batch_size, len(contents))]
                        batch_embeddings = list(dense_model.embed(batch))
                        all_dense_embeddings.extend(batch_embeddings)
                        dense_progress.progress(min(1.0, (i + batch_size) / len(contents)))
                    
                    st.info("Generating BM25 sparse embeddings...")
                    bm25_progress = st.progress(0)
                    
                    # Process sparse embeddings in batches
                    all_bm25_embeddings = []
                    for i in range(0, len(contents), batch_size):
                        batch = contents[i:min(i+batch_size, len(contents))]
                        for text in batch:
                            bm25_emb = next(bm25_model.embed(text))
                            all_bm25_embeddings.append(bm25_emb)
                        bm25_progress.progress(min(1.0, (i + batch_size) / len(contents)))
                    
                    st.info(f"✅ Generated embeddings for {len(contents)} chunks")
                    
                    # Prepare points for upload with both dense and sparse vectors
                    st.info("Preparing points for upload...")
                    points = []
                    for idx, (chunk, dense_emb, bm25_emb) in enumerate(zip(all_chunks, all_dense_embeddings, all_bm25_embeddings)):
                        point = models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector={
                                "dense-vector": dense_emb.tolist(),
                                "output-token-embeddings": dense_emb.tolist(),
                                "sparse": models.SparseVector(
                                    indices=bm25_emb.indices,
                                    values=bm25_emb.values
                                )
                            },
                            payload={
                                "text": chunk["content"],
                                "content": chunk["content"],
                                "chunk_id": chunk.get("chunk_id", str(idx)),
                                "page_num": chunk["page_num"],
                                "position": chunk.get("position", 0),
                                "is_page_start": chunk.get("is_page_start", False),
                                "is_page_end": chunk.get("is_page_end", False),
                                "source_type": chunk.get("source_type", "chunk")
                            }
                        )
                        points.append(point)
                else:
                    # Fallback to standard embedding if FastEmbed is not available
                    st.warning("FastEmbed not available. Using standard vector search only.")
                    
                    # Get dense embedding model
                    model = load_sentence_transformer()
                    
                    # Extract content from chunks
                    contents = [chunk["content"] for chunk in all_chunks]
                    
                    # Generate dense embeddings only with progress tracking
                    st.info("Generating embeddings...")
                    embedding_progress = st.progress(0)
                    
                    # Process in batches
                    batch_size = 20
                    embeddings = []
                    
                    for i in range(0, len(contents), batch_size):
                        batch = contents[i:min(i+batch_size, len(contents))]
                        batch_embeddings = model.encode(batch)
                        embeddings.extend(batch_embeddings)
                        embedding_progress.progress(min(1.0, (i + batch_size) / len(contents)))
                    
                    st.info(f"✅ Generated embeddings for {len(contents)} chunks")
                    
                    # Prepare points for upload with dense vectors
                    st.info("Preparing points for upload...")
                    points = []
                    for idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
                        embedding_list = embedding.tolist()
                        point = models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector={
                                "dense-vector": embedding_list,
                                "output-token-embeddings": embedding_list
                            },
                            payload={
                                "text": chunk["content"],
                                "content": chunk["content"],
                                "chunk_id": chunk.get("chunk_id", str(idx)),
                                "page_num": chunk["page_num"],
                                "position": chunk.get("position", 0),
                                "is_page_start": chunk.get("is_page_start", False),
                                "is_page_end": chunk.get("is_page_end", False),
                                "source_type": chunk.get("source_type", "chunk")
                            }
                        )
                        points.append(point)
                
                # Upload to collection with progress tracking
                st.info(f"Uploading {len(points)} points to Qdrant...")
                upload_progress = st.progress(0)
                
                # Upload in batches to show progress
                batch_size = 50
                for i in range(0, len(points), batch_size):
                    batch = points[i:min(i+batch_size, len(points))]
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=batch
                    )
                    upload_progress.progress(min(1.0, (i + batch_size) / len(points)))
                
                st.success(f"✅ Successfully uploaded {len(points)} points with hybrid search capabilities")
        else:
            # Non-interactive processing without progress container
            # Extract pages
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text.strip():  # Only process non-empty pages
                    # Clean the text
                    clean_page_text = clean_text(page_text)
                    
                    # Add page metadata
                    all_chunks.append({
                        "content": clean_page_text,
                        "page_num": page_num + 1,
                        "is_page_start": True,
                        "is_page_end": True,
                        "source_type": "page"
                    })
            
            # Split text for more granular chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            detailed_chunks = []
            for page_chunk in all_chunks:
                smaller_chunks = text_splitter.split_text(page_chunk["content"])
                for idx, chunk in enumerate(smaller_chunks):
                    detailed_chunks.append({
                        "content": chunk,
                        "chunk_id": f"p{page_chunk['page_num']}_c{idx}",
                        "page_num": page_chunk["page_num"],
                        "position": idx,
                        "is_page_start": idx == 0,
                        "is_page_end": idx == len(smaller_chunks) - 1,
                        "source_type": "chunk"
                    })
            
            # Combine with page-level chunks for multi-granularity search
            all_chunks.extend(detailed_chunks)
            
            # Get Qdrant client
            qdrant_client = setup_qdrant_client()
            
            # Check if collection exists and has points
            collection_info = qdrant_client.get_collection(collection_name)
            existing_count = collection_info.points_count
            
            # Skip if chunks are already added
            if existing_count > 0:
                return
            
            # Generate embeddings and upload to Qdrant
            if FASTEMBED_AVAILABLE and dense_model and bm25_model:
                # Using FastEmbed for hybrid search with dense and BM25 embeddings
                # Extract content from chunks
                contents = [chunk["content"] for chunk in all_chunks]
                
                # Generate embeddings
                all_dense_embeddings = list(dense_model.embed(contents))
                
                # Process sparse embeddings
                all_bm25_embeddings = []
                for text in contents:
                    bm25_emb = next(bm25_model.embed(text))
                    all_bm25_embeddings.append(bm25_emb)
                
                # Prepare points for upload with both dense and sparse vectors
                points = []
                for idx, (chunk, dense_emb, bm25_emb) in enumerate(zip(all_chunks, all_dense_embeddings, all_bm25_embeddings)):
                    point = models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense-vector": dense_emb.tolist(),
                            "output-token-embeddings": dense_emb.tolist(),
                            "sparse": models.SparseVector(
                                indices=bm25_emb.indices,
                                values=bm25_emb.values
                            )
                        },
                        payload={
                            "text": chunk["content"],
                            "content": chunk["content"],
                            "chunk_id": chunk.get("chunk_id", str(idx)),
                            "page_num": chunk["page_num"],
                            "position": chunk.get("position", 0),
                            "is_page_start": chunk.get("is_page_start", False),
                            "is_page_end": chunk.get("is_page_end", False),
                            "source_type": chunk.get("source_type", "chunk")
                        }
                    )
                    points.append(point)
            else:
                # Fallback to standard embedding if FastEmbed is not available
                # Get dense embedding model
                model = load_sentence_transformer()
                
                # Extract content from chunks
                contents = [chunk["content"] for chunk in all_chunks]
                
                # Generate dense embeddings only
                embeddings = model.encode(contents)
                
                # Prepare points for upload with dense vectors
                points = []
                for idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
                    embedding_list = embedding.tolist()
                    point = models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense-vector": embedding_list,
                            "output-token-embeddings": embedding_list
                        },
                        payload={
                            "text": chunk["content"],
                            "content": chunk["content"],
                            "chunk_id": chunk.get("chunk_id", str(idx)),
                            "page_num": chunk["page_num"],
                            "position": chunk.get("position", 0),
                            "is_page_start": chunk.get("is_page_start", False),
                            "is_page_end": chunk.get("is_page_end", False),
                            "source_type": chunk.get("source_type", "chunk")
                        }
                    )
                    points.append(point)
            
            # Upload to collection
            for i in range(0, len(points), 50):
                batch = points[i:min(i+50, len(points))]
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        raise e

# Search with BestRAG-style hybrid approach
def hybrid_search(collection_name, query_text, limit=4, use_reranking=True):
    """
    Perform hybrid search based on BestRAG implementation:
    1. Dense vector search
    2. Sparse vector (BM25) search 
    3. Late interaction search
    4. Optional reranking
    """
    try:
        # Clean query text
        cleaned_query = clean_text(query_text)
        
        # Get models
        if FASTEMBED_AVAILABLE:
            dense_model = load_dense_model()
            bm25_model = load_bm25_model()
            
            # Get Qdrant client
            qdrant_client = setup_qdrant_client()
            
            if dense_model and bm25_model:
                # Generate query embeddings
                dense_vector = next(dense_model.embed(cleaned_query)).tolist()
                bm25_vector = next(bm25_model.embed(cleaned_query))
                
                # Build the query
                query_vector = {
                    "dense-vector": dense_vector,
                    "output-token-embeddings": dense_vector,
                    "sparse": {
                        "indices": bm25_vector.indices,
                        "values": bm25_vector.values,
                    }
                }
                
                # Set up prefetch for hybrid search
                prefetch = [
                    models.Prefetch(
                        query=query_vector["dense-vector"],
                        using="dense-vector",
                        limit=int(limit * 1.5),
                    )
                ]
                
                # Perform hybrid search using query_points
                search_results = qdrant_client.query_points(
                    collection_name=collection_name,
                    prefetch=prefetch,
                    query=query_vector["output-token-embeddings"],
                    using="output-token-embeddings",
                    limit=limit * 2  # Get more results for reranking
                )
                
                # Perform a sparse-only search for diversity
                if "sparse" in query_vector:
                    sparse_results = qdrant_client.search(
                        collection_name=collection_name,
                        query_vector={"sparse": models.SparseVector(
                            indices=bm25_vector.indices,
                            values=bm25_vector.values
                        )},
                        limit=limit,
                        with_payload=True
                    )
                else:
                    sparse_results = []
                
                # Combine results (removing duplicates)
                all_results = {}  # Use dict to deduplicate
                
                # Process hybrid results
                for result in search_results:
                    result_id = result.id
                    all_results[result_id] = {
                        "text": result.payload["text"],
                        "score": result.score,
                        "page": result.payload.get("page_num", 0),
                        "source": "hybrid",
                        "combined_score": result.score
                    }
                
                # Process sparse results
                for result in sparse_results:
                    result_id = result.id
                    if result_id in all_results:
                        # Boost score for results found by multiple methods
                        all_results[result_id]["combined_score"] = max(
                            all_results[result_id]["combined_score"],
                            result.score
                        ) * 1.1  # Boost by 10%
                        all_results[result_id]["source"] += "+sparse"
                    else:
                        all_results[result_id] = {
                            "text": result.payload["text"],
                            "score": result.score,
                            "page": result.payload.get("page_num", 0),
                            "source": "sparse",
                            "combined_score": result.score
                        }
            else:
                # Fallback to standard search
                return fallback_search(collection_name, cleaned_query, limit, use_reranking)
        else:
            # Fallback to standard search
            return fallback_search(collection_name, cleaned_query, limit, use_reranking)
        
        # Simple reranking: favor chunks that appear in multiple search methods
        # and promote contextual coherence by considering page numbers
        final_results = list(all_results.values())
        
        if use_reranking:
            # Sort by combined score
            final_results.sort(key=lambda x: x["combined_score"], reverse=True)
            
            # Take top results ensuring page diversity if possible
            seen_pages = set()
            ranked_results = []
            
            # First add highest scoring result from each page
            for result in final_results:
                page = result["page"]
                if page not in seen_pages and len(ranked_results) < limit:
                    ranked_results.append(result)
                    seen_pages.add(page)
            
            # Then add remaining results by score
            remaining = [r for r in final_results if r not in ranked_results]
            remaining.sort(key=lambda x: x["combined_score"], reverse=True)
            
            ranked_results.extend(remaining[:limit-len(ranked_results)])
            final_results = ranked_results[:limit]
        else:
            # Just sort by score and take top results
            final_results.sort(key=lambda x: x["combined_score"], reverse=True)
            final_results = final_results[:limit]
        
        # Extract texts for final result
        result_texts = [result["text"] for result in final_results]
        result_metadata = final_results
        
        return result_texts, result_metadata
        
    except Exception as e:
        st.error(f"Error in hybrid search: {e}")
        return [], []

# Fallback search method if hybrid search is unavailable
def fallback_search(collection_name, query_text, limit=4, use_reranking=True):
    """Fallback search method using standard vector search and text filters"""
    try:
        # Get model and client
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        
        # Generate embedding for query
        query_embedding = model.encode(query_text).tolist()
        
        # 1. Pure vector search
        vector_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector={"dense-vector": query_embedding},
            limit=limit,
            with_payload=True
        )
        
        # 2. Text-based search (keyword matching)
        keyword_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="content",
                    match=models.MatchText(text=query_text)
                )
            ]
        )
        
        try:
            text_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector={"dense-vector": query_embedding},
                query_filter=keyword_filter,
                limit=limit,
                with_payload=True
            )
        except Exception as e:
            # Fallback to standard search if text filter fails
            text_results = []
        
        # 3. Page-level search (larger context chunks)
        page_filter = models.FieldCondition(
            key="source_type",
            match=models.MatchValue(value="page")
        )
        
        page_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector={"dense-vector": query_embedding},
            query_filter=models.Filter(
                must=[page_filter]
            ),
            limit=max(2, limit//2),  # Fewer but larger chunks
            with_payload=True
        )
        
        # Combine all results (removing duplicates)
        all_results = {}  # Use dict to deduplicate
        
        # Process and score results (add source information)
        for idx, result in enumerate(vector_results):
            result_id = result.id
            all_results[result_id] = {
                "text": result.payload["text"],
                "score": result.score,
                "page": result.payload.get("page_num", 0),
                "source": "vector",
                "combined_score": result.score
            }
        
        for idx, result in enumerate(text_results):
            result_id = result.id
            if result_id in all_results:
                # Boost score for results found by multiple methods
                all_results[result_id]["combined_score"] = max(
                    all_results[result_id]["combined_score"],
                    result.score
                ) * 1.1  # Boost by 10%
                all_results[result_id]["source"] += "+text"
            else:
                all_results[result_id] = {
                    "text": result.payload["text"],
                    "score": result.score,
                    "page": result.payload.get("page_num", 0),
                    "source": "text",
                    "combined_score": result.score
                }
        
        for idx, result in enumerate(page_results):
            result_id = result.id
            if result_id in all_results:
                # Boost score for results found by multiple methods
                all_results[result_id]["combined_score"] = max(
                    all_results[result_id]["combined_score"],
                    result.score
                ) * 1.1  # Boost by 10%
                all_results[result_id]["source"] += "+page"
            else:
                all_results[result_id] = {
                    "text": result.payload["text"],
                    "score": result.score * 0.95,  # Slightly discount page results
                    "page": result.payload.get("page_num", 0),
                    "source": "page",
                    "combined_score": result.score * 0.95
                }
        
        # Reranking if enabled
        final_results = list(all_results.values())
        
        if use_reranking:
            # Sort by combined score
            final_results.sort(key=lambda x: x["combined_score"], reverse=True)
            
            # Take top results ensuring page diversity if possible
            seen_pages = set()
            ranked_results = []
            
            # First add highest scoring result from each page
            for result in final_results:
                page = result["page"]
                if page not in seen_pages and len(ranked_results) < limit:
                    ranked_results.append(result)
                    seen_pages.add(page)
            
            # Then add remaining results by score
            remaining = [r for r in final_results if r not in ranked_results]
            remaining.sort(key=lambda x: x["combined_score"], reverse=True)
            
            ranked_results.extend(remaining[:limit-len(ranked_results)])
            final_results = ranked_results[:limit]
        else:
            # Just sort by score and take top results
            final_results.sort(key=lambda x: x["combined_score"], reverse=True)
            final_results = final_results[:limit]
        
        # Extract texts for final result
        result_texts = [result["text"] for result in final_results]
        result_metadata = final_results
        
        return result_texts, result_metadata
        
    except Exception as e:
        st.error(f"Error in fallback search: {e}")
        return [], []

# Ensemble search combining multiple retrieval methods
def ensemble_search(collection_name, query_text, limit=4, vector_weight=0.7, use_reranking=True):
    """
    Perform ensemble search combining multiple retrieval techniques:
    1. BestRAG-style hybrid search
    2. Pure vector search
    3. Text search
    4. Page-level context search
    5. Optional reranking
    """
    try:
        # Get hybrid search results
        hybrid_texts, hybrid_metadata = hybrid_search(
            collection_name=collection_name,
            query_text=query_text,
            limit=int(limit * 1.5),  # Get more for ensemble
            use_reranking=False  # We'll do reranking at the end
        )
        
        # Get fallback search results
        fallback_texts, fallback_metadata = fallback_search(
            collection_name=collection_name,
            query_text=query_text,
            limit=int(limit * 1.5),  # Get more for ensemble
            use_reranking=False  # We'll do reranking at the end
        )
        
        # Combine all results (removing duplicates)
        all_results = {}  # Use dict to deduplicate
        
        # Process hybrid results
        for result in hybrid_metadata:
            text = result["text"]
            text_hash = hashlib.md5(text.encode()).hexdigest()  # Use text hash as key for deduplication
            
            all_results[text_hash] = {
                "text": text,
                "score": result["score"] * vector_weight,  # Apply weight to hybrid results
                "page": result["page"],
                "source": result["source"],
                "combined_score": result["score"] * vector_weight
            }
        
        # Process fallback results
        for result in fallback_metadata:
            text = result["text"]
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            if text_hash in all_results:
                # Boost score for results found by multiple methods
                all_results[text_hash]["combined_score"] = max(
                    all_results[text_hash]["combined_score"],
                    result["score"] * (1 - vector_weight)
                ) * 1.1  # Boost by 10%
                all_results[text_hash]["source"] += "+" + result["source"]
            else:
                all_results[text_hash] = {
                    "text": text,
                    "score": result["score"] * (1 - vector_weight),  # Apply weight to fallback results
                    "page": result["page"],
                    "source": result["source"],
                    "combined_score": result["score"] * (1 - vector_weight)
                }
        
        # Apply reranking if enabled
        final_results = list(all_results.values())
        
        if use_reranking:
            # Sort by combined score
            final_results.sort(key=lambda x: x["combined_score"], reverse=True)
            
            # Take top results ensuring page diversity if possible
            seen_pages = set()
            ranked_results = []
            
            # First add highest scoring result from each page
            for result in final_results:
                page = result["page"]
                if page not in seen_pages and len(ranked_results) < limit:
                    ranked_results.append(result)
                    seen_pages.add(page)
            
            # Then add remaining results by score
            remaining = [r for r in final_results if r not in ranked_results]
            remaining.sort(key=lambda x: x["combined_score"], reverse=True)
            
            ranked_results.extend(remaining[:limit-len(ranked_results)])
            final_results = ranked_results[:limit]
        else:
            # Just sort by score and take top results
            final_results.sort(key=lambda x: x["combined_score"], reverse=True)
            final_results = final_results[:limit]
        
        # Extract texts for final result
        result_texts = [result["text"] for result in final_results]
        result_metadata = final_results
        
        return result_texts, result_metadata
        
    except Exception as e:
        st.error(f"Error in ensemble search: {e}")
        # Fallback to standard hybrid search
        return hybrid_search(collection_name, query_text, limit, use_reranking)

# Process PDF file
if uploaded_file is not None:
    # Calculate file hash for caching/collection
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    if USE_FAISS:
        # FAISS Implementation
        cache_file = VECTORDB_DIR / f"{file_hash}.pkl"
        
        # Check if we need to process the file
        if st.session_state.retriever is None:
            with st.spinner("Processing PDF with FAISS..."):
                # Load from cache if exists
                if cache_file.exists():
                    try:
                        with open(cache_file, "rb") as f:
                            vector_store = pickle.load(f)
                        st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                        st.success("Loaded from cache")
                    except Exception as e:
                        st.error(f"Error loading cache: {e}")
                        # Will proceed to regenerate
                
                # Process if not in cache or failed to load
                if st.session_state.retriever is None:
                    try:
                        # Extract text
                        reader = PdfReader(io.BytesIO(file_bytes))
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        
                        # Split text
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        chunks = text_splitter.split_text(text)
                        
                        # Create embeddings and vector store
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        )
                        
                        vector_store = FAISS.from_texts(chunks, embeddings)
                        
                        # Save to cache
                        with open(cache_file, "wb") as f:
                            pickle.dump(vector_store, f)
                        
                        # Create retriever
                        st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                        
                        st.success("PDF processed successfully with FAISS!")
                    except Exception as e:
                        st.error(f"Error processing PDF with FAISS: {e}")
    else:
        # Qdrant Implementation
        collection_name = f"pdf_{file_hash}"
        
        # Check if we need to process the file
        if st.session_state.collection_name != collection_name:
            with st.spinner("Processing PDF with Qdrant..."):
                # Setup Qdrant client
                qdrant_client = setup_qdrant_client()
                
                # Create expandable section to show processing details
                processing_expander = st.expander("View Processing Details", expanded=True)
                
                if qdrant_client:
                    # Check if collection already exists and has points
                    collections = qdrant_client.get_collections().collections
                    collection_exists = collection_name in [c.name for c in collections]
                    
                    if collection_exists:
                        try:
                            collection_info = qdrant_client.get_collection(collection_name)
                            existing_count = collection_info.points_count
                            
                            if existing_count > 0:
                                with processing_expander:
                                    st.success(f"✅ Found existing collection '{collection_name}' with {existing_count} points")
                                    st.info("Reusing cached vectors from previous processing")
                                    st.progress(100)
                                
                                # Update session state
                                st.session_state.collection_name = collection_name
                                st.success("PDF loaded from cache successfully!")
                                continue
                        except Exception as e:
                            with processing_expander:
                                st.error(f"Error checking collection: {e}")
                    
                    # Get model for vector size
                    model = load_sentence_transformer()
                    vector_size = model.get_sentence_embedding_dimension()
                    
                    # Create collection with appropriate vector size
                    with processing_expander:
                        st.info(f"Creating collection '{collection_name}'...")
                    
                    create_collection(qdrant_client, collection_name, vector_size)
                    
                    # Process PDF and add to collection
                    try:
                        process_pdf_qdrant(file_bytes, collection_name, processing_expander)
                        
                        # Update session state
                        st.session_state.collection_name = collection_name
                        st.success("PDF processed successfully with Qdrant!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
                else:
                    st.error("Failed to initialize Qdrant client")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.markdown(source)
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your PDF"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if USE_FAISS:
        # FAISS implementation for retrieval
        # Check if retriever is available
        if st.session_state.retriever is None:
            with st.chat_message("assistant"):
                st.markdown("Please upload a PDF file first.")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload a PDF file first."})
        else:
            # Get relevant documents
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant chunks
                    docs = st.session_state.retriever.get_relevant_documents(prompt)
                    sources = [doc.page_content for doc in docs]
                    
                    # Prepare context
                    context = "\n\n".join(sources)
                    
                    # Prepare prompt for LLM
                    full_prompt = f"""
                    Answer the following question based on the provided context.
                    
                    Context:
                    {context}
                    
                    Question: {prompt}
                    
                    Answer:
                    """
                    
                    # Get response from your custom LLM function
                    # This is a placeholder - replace with your actual LLM call
                    response = f"Using local LLM to answer: {prompt}\n\nBased on the document, I found relevant information that would help answer this question."
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(response)
                        
                        # Show sources with metadata
                        with st.expander("View Sources"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(source)
                                st.divider()
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    with st.chat_message("assistant"):
                        st.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        # Qdrant implementation for retrieval
        # Check if collection is available
        if st.session_state.collection_name is None:
            with st.chat_message("assistant"):
                st.markdown("Please upload a PDF file first.")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload a PDF file first."})
        else:
            # Get relevant documents
            with st.spinner("Thinking..."):
                try:
                    # Set up search parameters
                    vector_weight = st.session_state.get('hybrid_weight', 0.7)
                    limit = st.session_state.get('result_count', 4)
                    use_reranking = st.session_state.get('use_reranking', True)
                    use_ensemble = st.session_state.get('use_ensemble', True)
                    use_vector_only = st.session_state.get('use_vector_only', False)
                    use_hybrid_only = st.session_state.get('use_hybrid_only', False)
                    
                    # Retrieve relevant chunks using selected search method
                    if use_vector_only:
                        # Get model and client
                        model = load_sentence_transformer()
                        qdrant_client = setup_qdrant_client()
                        
                        # Generate embedding for query
                        query_embedding = model.encode(prompt).tolist()
                        
                        # Pure vector search
                        search_results = qdrant_client.search(
                            collection_name=st.session_state.collection_name,
                            query_vector={"dense-vector": query_embedding},
                            limit=limit,
                            with_payload=True
                        )
                        
                        # Extract results
                        sources = [result.payload["text"] for result in search_results]
                        metadata = [{
                            "text": result.payload["text"],
                            "score": result.score,
                            "page": result.payload.get("page_num", 0),
                            "source": "vector"
                        } for result in search_results]
                        
                    elif use_hybrid_only:
                        # Use hybrid search
                        sources, metadata = hybrid_search(
                            collection_name=st.session_state.collection_name,
                            query_text=prompt,
                            limit=limit,
                            use_reranking=use_reranking
                        )
                    else:
                        # Use ensemble search (default)
                        sources, metadata = ensemble_search(
                            collection_name=st.session_state.collection_name,
                            query_text=prompt,
                            limit=limit,
                            vector_weight=vector_weight,
                            use_reranking=use_reranking
                        )
                    
                    if not sources:
                        with st.chat_message("assistant"):
                            st.markdown("I couldn't find relevant information in the document to answer your question. Please try a different question or upload a different PDF.")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "I couldn't find relevant information in the document to answer your question. Please try a different question or upload a different PDF."
                        })
                    else:
                        # Prepare context
                        context = "\n\n".join(sources)
                        
                        # Prepare prompt for LLM
                        full_prompt = f"""
                        Answer the following question based on the provided context.
                        
                        Context:
                        {context}
                        
                        Question: {prompt}
                        
                        Answer:
                        """
                        
                        # Get response
                        # This is a placeholder - replace with your actual LLM call
                        response = f"I've analyzed the document and found relevant information to answer: {prompt}\n\nTo implement a complete solution, you would need to connect your preferred LLM here."
                        
                        # Display assistant response
                        with st.chat_message("assistant"):
                            st.markdown(response)
                            
                            # Show sources with metadata
                            with st.expander("View Sources"):
                                for i, source in enumerate(sources):
                                    meta = metadata[i] if i < len(metadata) else {}
                                    source_type = meta.get("source", "")
                                    score = meta.get("score", 0)
                                    page = meta.get("page", 0)
                                    
                                    st.markdown(f"**Source {i+1}:** (Page {page}, Score: {score:.4f}, Method: {source_type})")
                                    st.markdown(source)
                                    st.divider()
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                    
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    with st.chat_message("assistant"):
                        st.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
