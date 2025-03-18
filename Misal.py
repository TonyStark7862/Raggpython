# Add necessary imports at the top
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import pickle
import hashlib
import os
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


# Flag to determine which vector database to use
USE_FAISS = False  # Set to False to use Qdrant instead

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
    """Create a new collection if it doesn't exist with advanced search capabilities."""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            # Create collection with optimized configuration for advanced search
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size, 
                    distance=Distance.COSINE,
                    # Add optimal parameters for semantic search
                    hnsw_config=models.HnswConfigDiff(
                        m=16,  # Number of bidirectional links for each node
                        ef_construct=100,  # Number of neighbors to consider during index construction
                        full_scan_threshold=10000  # When to switch to full scan from index
                    )
                ),
                # Enable optimizers for better performance
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=20000,  # Threshold for using memory mapping
                    indexing_threshold=20000  # Threshold for using index vs full scan
                ),
                # Set up sparse vectors for text search component
                sparse_vectors_config={
                    "text": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            full_scan_threshold=10000  # When to switch to full scan from index
                        )
                    )
                },
                # Add shard number based on expected size
                shard_number=1  # Increase for larger collections
            )
            st.success(f"✅ Collection '{collection_name}' created with advanced search capabilities")
        else:
            st.info(f"Using existing collection '{collection_name}'")
    except Exception as e:
        st.error(f"Error creating collection: {e}")

# Process PDF and add to Qdrant with hybrid search support
def process_pdf_qdrant(file_bytes, collection_name, progress_container=None):
    try:
        # Extract text
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        all_chunks = []
        
        # Skip extraction if we already did it (for cases where progress_container is passed)
        if progress_container is None:
            # Process each page with page number tracking
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text.strip():  # Only process non-empty pages
                    # Add page metadata
                    all_chunks.append({
                        "content": page_text,
                        "page_num": page_num + 1,
                        "is_page_start": True,
                        "is_page_end": True,
                        "source_type": "page"
                    })
                    text += page_text + "\n"
            
            # Split text for more granular chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(text)
        else:
            # We've already extracted and split the text
            chunks = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text.strip():  # Only process non-empty pages
                    # Add page metadata
                    all_chunks.append({
                        "content": page_text,
                        "page_num": page_num + 1,
                        "is_page_start": True,
                        "is_page_end": True,
                        "source_type": "page"
                    })
        
        # Add metadata to chunks
        detailed_chunks = []
        for idx, chunk in enumerate(chunks):
            # Determine which page this chunk likely belongs to
            page_text_pos = text.find(chunk[:100])  # Use first 100 chars to locate
            page_num = 1
            cumulative_length = 0
            
            # Simple heuristic to determine page number
            for p_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                cumulative_length += len(page_text)
                if page_text_pos <= cumulative_length:
                    page_num = p_num + 1
                    break
            
            detailed_chunks.append({
                "content": chunk,
                "chunk_id": idx,
                "page_num": page_num,
                "position": idx,
                "is_page_start": False,
                "is_page_end": False,
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
            if progress_container:
                with progress_container:
                    st.info(f"Document chunks already added to collection (found {existing_count} points)")
            else:
                st.info(f"Document chunks already added to collection (found {existing_count} points)")
            return
        
        # Generate embeddings based on available models
        if FASTEMBED_AVAILABLE:
            if progress_container:
                with progress_container:
                    st.info("Using FastEmbed for hybrid search with dense and BM25 embeddings")
                    
                    # Initialize models
                    dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
                    bm25_model = SparseTextEmbedding("Qdrant/bm25")
                    
                    # Extract content from chunks
                    contents = [chunk["content"] for chunk in all_chunks]
                    
                    # Generate embeddings
                    st.info("Generating dense embeddings...")
                    dense_progress = st.progress(0)
                    dense_embeddings = list(dense_model.embed(contents))
                    dense_progress.progress(1.0)
                    
                    st.info("Generating BM25 sparse embeddings...")
                    bm25_progress = st.progress(0)
                    bm25_embeddings = list(bm25_model.embed(contents))
                    bm25_progress.progress(1.0)
                    
                    st.info(f"✅ Generated embeddings for {len(contents)} chunks")
                    
                    # Prepare points for upload with both dense and sparse vectors
                    st.info("Preparing points for upload...")
                    points = []
                    for idx, (chunk, dense_emb, bm25_emb) in enumerate(zip(all_chunks, dense_embeddings, bm25_embeddings)):
                        point = models.PointStruct(
                            id=idx,
                            vector={
                                "dense": dense_emb,
                                "bm25": models.SparseVector(**bm25_emb.as_object())
                            },
                            payload={
                                "text": chunk["content"],
                                "chunk_id": idx,
                                "content": chunk["content"],
                                "page_num": chunk["page_num"],
                                "position": idx,
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
                    batch_size = 100
                    for i in range(0, len(points), batch_size):
                        batch = points[i:min(i+batch_size, len(points))]
                        qdrant_client.upsert(
                            collection_name=collection_name,
                            points=batch
                        )
                        upload_progress.progress(min(1.0, (i + batch_size) / len(points)))
                    
                    st.success(f"✅ Successfully uploaded {len(points)} points with hybrid search capabilities")
                    
                    # Configure payload indexing for text search
                    st.info("Setting up additional payload indexes...")
                    
                    try:
                        # Index page numbers for filtering
                        qdrant_client.create_payload_index(
                            collection_name=collection_name,
                            field_name="page_num",
                            field_schema=models.PayloadSchemaType.INTEGER
                        )
                        st.success("✅ Index created for page numbers")
                    except Exception as e:
                        st.warning(f"Page number index creation failed (may already exist): {e}")
                    
                    try:
                        # Index source_type for filtering
                        qdrant_client.create_payload_index(
                            collection_name=collection_name,
                            field_name="source_type",
                            field_schema=models.PayloadSchemaType.KEYWORD
                        )
                        st.success("✅ Index created for source types")
                    except Exception as e:
                        st.warning(f"Source type index creation failed (may already exist): {e}")
                    
                    st.success("✅ Collection fully indexed with hybrid search support!")
            else:
                # Same code but without progress container
                # Initialize models
                dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
                bm25_model = SparseTextEmbedding("Qdrant/bm25")
                
                # Extract content from chunks
                contents = [chunk["content"] for chunk in all_chunks]
                
                # Generate embeddings
                dense_embeddings = list(dense_model.embed(contents))
                bm25_embeddings = list(bm25_model.embed(contents))
                
                # Prepare points for upload with both dense and sparse vectors
                points = []
                for idx, (chunk, dense_emb, bm25_emb) in enumerate(zip(all_chunks, dense_embeddings, bm25_embeddings)):
                    point = models.PointStruct(
                        id=idx,
                        vector={
                            "dense": dense_emb,
                            "bm25": models.SparseVector(**bm25_emb.as_object())
                        },
                        payload={
                            "text": chunk["content"],
                            "chunk_id": idx,
                            "content": chunk["content"],
                            "page_num": chunk["page_num"],
                            "position": idx,
                            "is_page_start": chunk.get("is_page_start", False),
                            "is_page_end": chunk.get("is_page_end", False),
                            "source_type": chunk.get("source_type", "chunk")
                        }
                    )
                    points.append(point)
                
                # Upload to collection
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                # Configure payload indexing
                try:
                    qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name="page_num",
                        field_schema=models.PayloadSchemaType.INTEGER
                    )
                except Exception:
                    # Index may already exist, continue
                    pass
                
                try:
                    qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name="source_type",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                except Exception:
                    # Index may already exist, continue
                    pass
                
                st.success(f"✅ Added {len(points)} chunks with hybrid search support")
        else:
            # Fallback to standard embedding if FastEmbed is not available
            # Get dense embedding model
            model = load_sentence_transformer()
            
            if progress_container:
                with progress_container:
                    st.warning("FastEmbed not available. Using standard vector search only.")
                    
                    # Extract content from chunks
                    contents = [chunk["content"] for chunk in all_chunks]
                    
                    # Generate dense embeddings only
                    st.info("Generating embeddings...")
                    embedding_progress = st.progress(0)
                    
                    # Create embeddings in batches to show progress
                    embeddings = []
                    batch_size = 10
                    for i in range(0, len(contents), batch_size):
                        batch = contents[i:i+batch_size]
                        batch_embeddings = model.encode(batch)
                        embeddings.extend(batch_embeddings)
                        embedding_progress.progress(min(1.0, (i + batch_size) / len(contents)))
                    
                    st.info(f"✅ Generated {len(embeddings)} embeddings")
                    
                    # Prepare points for upload - include rich metadata
                    points = [
                        models.PointStruct(
                            id=idx,
                            vector=embedding.tolist(),
                            payload={
                                "text": chunk["content"],  # For retrieval
                                "chunk_id": idx,
                                "content": chunk["content"],  # For text search
                                "page_num": chunk["page_num"],
                                "position": idx,
                                "is_page_start": chunk.get("is_page_start", False),
                                "is_page_end": chunk.get("is_page_end", False),
                                "source_type": chunk.get("source_type", "chunk")
                            }
                        )
                        for idx, (embedding, chunk) in enumerate(zip(embeddings, all_chunks))
                    ]
                    
                    # Upload to collection
                    st.info(f"Uploading {len(points)} points to Qdrant...")
                    upload_progress = st.progress(0)
                    
                    # Upload in batches to show progress
                    batch_size = 100
                    for i in range(0, len(points), batch_size):
                        batch = points[i:min(i+batch_size, len(points))]
                        qdrant_client.upsert(
                            collection_name=collection_name,
                            points=batch
                        )
                        upload_progress.progress(min(1.0, (i + batch_size) / len(points)))
                    
                    st.info(f"✅ Successfully uploaded {len(points)} points to Qdrant")
                    
                    # Configure payload indexing for text search
                    st.info("Setting up text indexing for search...")
                    
                    # Create text index for content field
                    try:
                        qdrant_client.create_payload_index(
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
                        st.success("✅ Text index created for content field")
                    except Exception as e:
                        st.warning(f"Text index creation failed (may already exist): {e}")
                    
                    try:
                        # Index page numbers for filtering
                        qdrant_client.create_payload_index(
                            collection_name=collection_name,
                            field_name="page_num",
                            field_schema=models.PayloadSchemaType.INTEGER
                        )
                        st.success("✅ Index created for page numbers")
                    except Exception as e:
                        st.warning(f"Page number index creation failed (may already exist): {e}")
                    
                    try:
                        # Index source_type for filtering
                        qdrant_client.create_payload_index(
                            collection_name=collection_name,
                            field_name="source_type",
                            field_schema=models.PayloadSchemaType.KEYWORD
                        )
                        st.success("✅ Index created for source types")
                    except Exception as e:
                        st.warning(f"Source type index creation failed (may already exist): {e}")
                    
                    st.success("✅ Collection fully indexed and ready to query!")
            else:
                # Standard embedding without progress container
                # Extract content from chunks
                contents = [chunk["content"] for chunk in all_chunks]
                
                # Generate embeddings
                embeddings = model.encode(contents)
                
                # Prepare points for upload - include rich metadata
                points = [
                    models.PointStruct(
                        id=idx,
                        vector=embedding.tolist(),
                        payload={
                            "text": chunk["content"],  # For retrieval
                            "chunk_id": idx,
                            "content": chunk["content"],  # For text search
                            "page_num": chunk["page_num"],
                            "position": idx,
                            "is_page_start": chunk.get("is_page_start", False),
                            "is_page_end": chunk.get("is_page_end", False),
                            "source_type": chunk.get("source_type", "chunk")
                        }
                    )
                    for idx, (embedding, chunk) in enumerate(zip(embeddings, all_chunks))
                ]
                
                # Upload to collection
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                # Configure payload indexing for advanced search
                try:
                    qdrant_client.create_payload_index(
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
                    # Index page numbers for filtering
                    qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name="page_num",
                        field_schema=models.PayloadSchemaType.INTEGER
                    )
                except Exception:
                    # Index may already exist, continue
                    pass
                
                try:
                    # Index source_type for filtering
                    qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name="source_type",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                except Exception:
                    # Index may already exist, continue
                    pass
                
                st.success(f"✅ Added {len(points)} chunks to collection with text indexing")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        raise e

# Ensemble search with multiple Qdrant retrieval methods
def ensemble_search(collection_name, query_text, limit=4, vector_weight=0.7, use_reranking=True, progress_callback=None):
    """
    Perform ensemble search combining multiple retrieval techniques:
    1. Pure vector search (semantic similarity)
    2. BM25 sparse vector search (if available)
    3. Contextual search (with page-level chunks)
    4. Optional reranking
    """
    try:
        # Get model and client
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback("Loading models and initializing search...")
        
        # Generate embedding for query
        if progress_callback:
            progress_callback("Encoding your query...")
        
        query_embedding = model.encode([query_text])[0]
        
        if FASTEMBED_AVAILABLE:
            # Using FastEmbed for hybrid search
            if progress_callback:
                progress_callback("Using FastEmbed for enhanced hybrid search...")
                
            try:
                # Initialize models
                if progress_callback:
                    progress_callback("Loading embedding models...")
                
                dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
                bm25_model = SparseTextEmbedding("Qdrant/bm25")
                
                # Generate query embeddings
                if progress_callback:
                    progress_callback("Generating dense and sparse embeddings...")
                
                dense_vector = next(dense_model.query_embed(query_text))
                bm25_vector = next(bm25_model.query_embed(query_text))
                
                # Set up prefetch for hybrid search
                prefetch = [
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=int(limit * 1.5),  # Get more results for better reranking
                    ),
                    models.Prefetch(
                        query=models.SparseVector(**bm25_vector.as_object()),
                        using="bm25",
                        limit=int(limit * 1.5),
                    ),
                ]
                
                # Perform hybrid search with prefetch
                if progress_callback:
                    progress_callback("Executing hybrid search with BM25 and dense vectors...")
                
                hybrid_results = qdrant_client.query_points(
                    collection_name=collection_name,
                    prefetch=prefetch,
                    query=dense_vector,  # Main search with dense vector
                    using="dense",  # Using dense as main search strategy
                    with_payload=True,
                    limit=int(limit * 2)  # Get more results for reranking
                )
                
                # Also perform vector-only search for comparison
                if progress_callback:
                    progress_callback("Performing vector-only search for comparison...")
                
                vector_results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=dense_vector,
                    limit=limit,
                    score_threshold=0.5,  # Filter low relevance results
                    with_payload=True
                )
                
                # Perform page-level search for context
                page_filter = models.FieldCondition(
                    key="source_type",
                    match=models.MatchValue(value="page")
                )
                
                page_results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=dense_vector,
                    query_filter=models.Filter(
                        must=[page_filter]
                    ),
                    limit=max(2, limit//2),  # Fewer but larger chunks
                    with_payload=True
                )
                
                # Convert hybrid results to expected format
                hybrid_results_formatted = []
                for result in hybrid_results:
                    hybrid_results_formatted.append(models.ScoredPoint(
                        id=result.id,
                        version=0,
                        score=result.score,
                        payload=result.payload,
                        vector=None
                    ))
                
                # Now combine results as before
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
                
                for idx, result in enumerate(hybrid_results_formatted):
                    result_id = result.id
                    if result_id in all_results:
                        # Boost score for results found by multiple methods
                        all_results[result_id]["combined_score"] = max(
                            all_results[result_id]["combined_score"],
                            result.score
                        ) * 1.1  # Boost by 10%
                        all_results[result_id]["source"] += "+hybrid"
                    else:
                        all_results[result_id] = {
                            "text": result.payload["text"],
                            "score": result.score,
                            "page": result.payload.get("page_num", 0),
                            "source": "hybrid",
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
                            "text": result.payloa
