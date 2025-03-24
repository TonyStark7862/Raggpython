import os
import re
import uuid
import tempfile
import logging
import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import fitz  # PyMuPDF for advanced PDF handling
import tabula  # For table extraction
from typing import List, Dict, Tuple, Optional, Union, Any

# Initialize FastEmbed for embeddings as shown in documentation
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding

# Qdrant client for vector database
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue, SparseVector
from qdrant_client.models import VectorParams, Distance

# Text processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
import base64
from io import BytesIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
VECTOR_DB_PATH = "./vector_db"
COLLECTION_NAME = "hybrid-search"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
PDF_FOLDER = "./pdfs"

# Create necessary directories
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

# Initialize embedding models as shown in the documentation
@st.cache_resource
def load_embedding_models():
    # Following the exact model initialization from documentation
    dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    bm25_embedding_model = SparseTextEmbedding("qdrant/bm25")
    late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    
    return {
        "dense": dense_embedding_model,
        "sparse": bm25_embedding_model,
        "late": late_interaction_embedding_model
    }

# Initialize Qdrant client
@st.cache_resource
def get_qdrant_client():
    # Initialize client with local path as shown in documentation
    client = QdrantClient(path=VECTOR_DB_PATH)
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if COLLECTION_NAME not in collection_names:
        # Create collection with multi-vector setup as shown in documentation
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "all-MiniLM-L6-v2": VectorParams(size=384, distance=Distance.COSINE),
                "bm25": VectorParams(size=65536, distance=Distance.JACCARD),
                "colbertv2.0": VectorParams(size=128, distance=Distance.COSINE),
            },
            on_disk_payload=True
        )
        logger.info(f"Created new collection: {COLLECTION_NAME}")
    else:
        logger.info(f"Collection {COLLECTION_NAME} already exists")
    
    return client

# Extract tables from PDF
def extract_tables_from_page(pdf_path, page_num):
    try:
        tables = tabula.read_pdf(pdf_path, pages=page_num+1, multiple_tables=True)
        if tables and len(tables) > 0:
            return tables
    except Exception as e:
        logger.warning(f"Error extracting tables: {str(e)}")
    return []

# Process PDF with special handling for tables
def process_pdf(pdf_path):
    """Process PDF with special handling for tables as requested"""
    documents = []
    try:
        # Using PyMuPDF for better PDF processing
        pdf_document = fitz.open(pdf_path)
        file_name = os.path.basename(pdf_path)
        total_pages = len(pdf_document)
        
        # Setup progress bar in UI
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for page_num, page in enumerate(pdf_document):
            # Update progress
            progress_percent = (page_num + 1) / total_pages
            progress_text.text(f"Processing page {page_num+1}/{total_pages}")
            progress_bar.progress(progress_percent)
            
            # Extract text content
            text = page.get_text()
            
            # Check for tables and handle them
            tables = extract_tables_from_page(pdf_path, page_num)
            tables_text = []
            
            if tables:
                for i, table in enumerate(tables):
                    if not table.empty:
                        table_str = f"\nTABLE {i+1}:\n{table.to_string()}\n"
                        tables_text.append(table_str)
                
                # For long tables, keep them together with surrounding context
                if tables_text:
                    tables_combined = "\n".join(tables_text)
                    # Ensure tables stay with relevant text
                    text = re.sub(r'(Table\s+\d+[.\:]*)', r'\1' + tables_combined, text, flags=re.IGNORECASE)
            
            # Create document dict with metadata
            doc = {
                "content": text,
                "metadata": {
                    "source": file_name,
                    "page": page_num + 1,
                    "total_pages": total_pages,
                    "has_tables": len(tables) > 0,
                }
            }
            documents.append(doc)
        
        # Clear progress indicators
        progress_text.empty()
        progress_bar.empty()
        
        logger.info(f"Processed {total_pages} pages from PDF: {file_name}")
        return documents
    
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")
        return []

# Create chunks with proper context preservation
def create_chunks(documents, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    chunks = []
    
    # Use RecursiveCharacterTextSplitter to preserve context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Process each document
    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        
        # Special handling for table-containing text to keep tables intact
        if metadata.get("has_tables", False):
            # Find table sections
            table_sections = re.finditer(r'(TABLE \d+:[\s\S]*?)(?=TABLE \d+:|$)', content)
            table_positions = []
            
            for match in table_sections:
                start, end = match.span()
                table_positions.append((start, end))
            
            if table_positions:
                # Process text between and around tables
                last_end = 0
                for start, end in table_positions:
                    # Text before table
                    if start > last_end:
                        before_text = content[last_end:start]
                        text_chunks = text_splitter.create_documents([before_text])
                        for i, chunk in enumerate(text_chunks):
                            chunks.append({
                                "content": chunk.page_content,
                                "metadata": {
                                    **metadata,
                                    "chunk_index": len(chunks),
                                    "chunk_type": "text"
                                }
                            })
                    
                    # The table itself (keep intact)
                    table_text = content[start:end]
                    chunks.append({
                        "content": table_text,
                        "metadata": {
                            **metadata,
                            "chunk_index": len(chunks),
                            "chunk_type": "table"
                        }
                    })
                    last_end = end
                
                # Text after the last table
                if last_end < len(content):
                    after_text = content[last_end:]
                    text_chunks = text_splitter.create_documents([after_text])
                    for i, chunk in enumerate(text_chunks):
                        chunks.append({
                            "content": chunk.page_content,
                            "metadata": {
                                **metadata,
                                "chunk_index": len(chunks),
                                "chunk_type": "text"
                            }
                        })
            else:
                # No tables found despite metadata flag, process normally
                text_chunks = text_splitter.create_documents([content])
                for i, chunk in enumerate(text_chunks):
                    chunks.append({
                        "content": chunk.page_content,
                        "metadata": {
                            **metadata,
                            "chunk_index": i,
                            "chunk_type": "text"
                        }
                    })
        else:
            # Normal document without tables
            text_chunks = text_splitter.create_documents([content])
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "content": chunk.page_content,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "chunk_type": "text"
                    }
                })
    
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

# Index documents to Qdrant following documentation approach
def index_documents(chunks, embedding_models, qdrant_client):
    # Setup progress tracking
    total_chunks = len(chunks)
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Following the documentation's approach for indexing with multiple vector types
    points = []
    
    # Process each chunk
    for idx, chunk in enumerate(chunks):
        # Update progress
        progress_percent = (idx + 1) / total_chunks
        progress_text.text(f"Indexing chunk {idx+1}/{total_chunks}")
        progress_bar.progress(progress_percent)
        
        content = chunk["content"]
        
        try:
            # Generate embeddings with each model as shown in documentation
            dense_embedding = embedding_models["dense"].embed(content)
            bm25_embedding = embedding_models["sparse"].embed(content)
            late_interaction_embedding = embedding_models["late"].embed(content)
            
            # Create point with multiple vectors and payload
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "all-MiniLM-L6-v2": dense_embedding,
                    "bm25": bm25_embedding.as_object(),  # Convert sparse vector to proper format
                    "colbertv2.0": late_interaction_embedding,
                },
                payload={
                    "document": chunk
                }
            )
            points.append(point)
            
            # Upload in batches of 100
            if len(points) >= 100 or idx == total_chunks - 1:
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                points = []  # Reset for next batch
                
        except Exception as e:
            logger.error(f"Error indexing chunk {idx}: {str(e)}")
            continue
    
    # Clear progress indicators
    progress_text.empty()
    progress_bar.empty()
    
    # Get collection info after indexing
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    logger.info(f"Indexed {total_chunks} chunks. Collection now has {collection_info.vectors_count} vectors.")
    
    return True

# Query function implementing the hybrid search and reranking from documentation
def query_documents(query, embedding_models, qdrant_client, top_k=5, page_filter=None):
    try:
        # Generate query embeddings as shown in documentation
        dense_vectors = next(embedding_models["dense"].query_embed(query))
        sparse_vectors = next(embedding_models["sparse"].query_embed(query))
        late_vectors = next(embedding_models["late"].query_embed(query))
        
        # Set up filter if page is specified
        filter_condition = None
        if page_filter:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document.metadata.page",
                        match=MatchValue(value=page_filter)
                    )
                ]
            )
        
        # Set up prefetch for hybrid search as shown in documentation
        prefetch = [
            models.Prefetch(
                query=dense_vectors,
                using="all-MiniLM-L6-v2",
                limit=20,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_vectors.as_object()),
                using="bm25",
                limit=20,
            )
        ]
        
        # Execute the hybrid search with reranking using late interaction embeddings
        # This follows exactly the documentation approach
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetch,
            query=late_vectors,
            using="colbertv2.0",
            with_payload=True,
            limit=top_k,
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.payload["document"]["content"],
                "metadata": result.payload["document"]["metadata"],
                "score": result.score
            })
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        return []

# Streamlit UI with professional design
def create_ui():
    # Configure page
    st.set_page_config(
        page_title="PDF RAG System", 
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional UI
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background-color: #f8f9fa;
        padding: 1.5rem;
    }
    
    /* Header styling */
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3a5a9b;
        margin-bottom: 1rem;
    }
    
    .subheader {
        font-size: 1.4rem;
        color: #555;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    
    /* Cards and containers */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #ddd;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #fbfbfb;
    }
    
    /* Progress indicators */
    .progress-label {
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    /* Processing info */
    .processing-step {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    
    .step-number {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background-color: #3a5a9b;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Results area */
    .result-item {
        border-left: 4px solid #3a5a9b;
        padding-left: 1rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
    
    .result-score {
        font-size: 0.8rem;
        color: #888;
        margin-bottom: 0.5rem;
    }
    
    .result-content {
        margin-bottom: 0.5rem;
    }
    
    .result-metadata {
        font-size: 0.8rem;
        color: #666;
        padding: 0.5rem;
        background-color: #eee;
        border-radius: 4px;
    }
    
    /* Buttons */
    .primary-button {
        background-color: #3a5a9b;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 5px;
        font-weight: 500;
        cursor: pointer;
    }
    
    .primary-button:hover {
        background-color: #2c4577;
    }
    
    /* Inputs */
    .query-input {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 0.75rem;
        width: 100%;
        font-size: 1rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-ready {
        background-color: #28a745;
    }
    
    .status-processing {
        background-color: #ffc107;
    }
    
    .status-empty {
        background-color: #dc3545;
    }
    
    /* Tables */
    table {
        width: 100%;
        border-collapse: collapse;
    }
    
    th, td {
        padding: 0.75rem;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    
    th {
        background-color: #f8f9fa;
        font-weight: 600;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 5px 5px 0 0;
        border-bottom: none;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3a5a9b;
        color: white;
    }
    
    /* Dropdown indicators */
    .dropdown-indicator {
        cursor: pointer;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .dropdown-indicator:hover {
        background-color: #eef1f5;
    }
    
    /* Document entries */
    .document-entry {
        display: flex;
        justify-content: space-between;
        padding: 12px;
        border-bottom: 1px solid #eee;
        align-items: center;
    }
    
    .document-name {
        font-weight: 500;
    }
    
    .document-meta {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Loader animation */
    @keyframes pulse {
        0% {
            opacity: 1;
        }
        50% {
            opacity: 0.3;
        }
        100% {
            opacity: 1;
        }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown('<div class="header-title">📚 Multi-PDF RAG System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">A production-grade document retrieval system with Qdrant</div>', unsafe_allow_html=True)
    
    return

# Function to display PDF processing workflow
def display_processing_steps():
    st.markdown("""
    <div class="card">
        <h3>Processing Workflow</h3>
        <div class="processing-step">
            <div class="step-number">1</div>
            <div>PDF Upload - Files are uploaded and saved locally</div>
        </div>
        <div class="processing-step">
            <div class="step-number">2</div>
            <div>Text Extraction - PDF content including tables and structure is extracted</div>
        </div>
        <div class="processing-step">
            <div class="step-number">3</div>
            <div>Chunking - Documents are split into semantic chunks while preserving context</div>
        </div>
        <div class="processing-step">
            <div class="step-number">4</div>
            <div>Embedding - Multiple embedding models (dense, sparse, late interaction) process each chunk</div>
        </div>
        <div class="processing-step">
            <div class="step-number">5</div>
            <div>Indexing - Embeddings and metadata are stored in Qdrant vector database</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Function to display query workflow
def display_query_workflow():
    st.markdown("""
    <div class="card">
        <h3>Query Workflow</h3>
        <div class="processing-step">
            <div class="step-number">1</div>
            <div>Query Embedding - User query is converted to multiple embedding types</div>
        </div>
        <div class="processing-step">
            <div class="step-number">2</div>
            <div>Hybrid Search - Combination of dense and sparse vectors for better retrieval</div>
        </div>
        <div class="processing-step">
            <div class="step-number">3</div>
            <div>Reranking - Late interaction model improves result relevance</div>
        </div>
        <div class="processing-step">
            <div class="step-number">4</div>
            <div>Results - Top matches are returned with relevance scores</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main application
def main():
    # Setup the UI
    create_ui()
    
    # Initialize embedding models and Qdrant client
    embedding_models = load_embedding_models()
    qdrant_client = get_qdrant_client()
    
    # Sidebar with configuration settings
    with st.sidebar:
        st.header("📋 System Configuration")
        
        # Vector DB status
        try:
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            vectors_count = collection_info.vectors_count
            status_color = "status-ready" if vectors_count > 0 else "status-empty"
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div class="status-indicator {status_color}"></div>
                <div>Vector DB Status: {'Ready' if vectors_count > 0 else 'Empty'}</div>
            </div>
            <div>Indexed Vectors: {vectors_count}</div>
            """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div class="status-indicator status-empty"></div>
                <div>Vector DB Status: Not Initialized</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Chunking settings
        st.subheader("Chunking Settings")
        chunk_size = st.slider("Chunk Size", 500, 2000, DEFAULT_CHUNK_SIZE, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, DEFAULT_CHUNK_OVERLAP, 50)
        
        # Advanced settings
        st.subheader("Advanced Settings")
        handle_tables = st.checkbox("Special table handling", value=True)
        st.caption("Keep tables intact during chunking")
        
        preserve_structure = st.checkbox("Preserve document structure", value=True)
        st.caption("Maintain headings and section context")
        
        # Model info
        st.subheader("Models Used")
        st.markdown("""
        • **Dense**: all-MiniLM-L6-v2
        • **Sparse**: BM25 (Qdrant)
        • **Late Interaction**: ColBERT v2.0
        """)
        
        # Memory usage 
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.metric("GPU Memory", f"{gpu_memory:.2f} GB")
        
        # System disk usage
        if os.path.exists(VECTOR_DB_PATH):
            total_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                          for dirpath, _, filenames in os.walk(VECTOR_DB_PATH) 
                          for filename in filenames)
            st.metric("Vector DB Size", f"{total_size/1e6:.2f} MB")
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["📥 Upload & Process", "🔍 Query Documents", "📊 Manage Documents"])
    
    # Tab 1: Upload & Process
    with tab1:
        st.markdown('<h2>Upload & Process Documents</h2>', unsafe_allow_html=True)
        
        # Display processing workflow
        display_processing_steps()
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to process and index"
        )
        
        if uploaded_files:
            st.markdown(f"<div>Selected {len(uploaded_files)} files</div>", unsafe_allow_html=True)
            
            # Process button
            if st.button("Process & Index Documents", key="process_btn"):
                # Process each PDF
                with st.expander("Processing Status", expanded=True):
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Save file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_path = tmp_file.name
                        
                        try:
                            # Create header for this file
                            st.markdown(f"### Processing: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                            
                            # 1. Process PDF
                            st.markdown("#### 📄 Extracting text and tables")
                            documents = process_pdf(temp_path)
                            
                            if documents:
                                # 2. Create chunks
                                st.markdown("#### ✂️ Creating chunks")
                                chunks = create_chunks(documents, chunk_size, chunk_overlap)
                                
                                # 3. Index to Qdrant
                                st.markdown("#### 🔢 Generating embeddings and indexing")
                                success = index_documents(chunks, embedding_models, qdrant_client)
                                
                                if success:
                                    # Save the file to persistent storage
                                    save_path = os.path.join(PDF_FOLDER, uploaded_file.name)
                                    with open(save_path, "wb") as f:
                                        f.write(uploaded_file.getvalue())
                                    
                                    st.success(f"Successfully processed and indexed {uploaded_file.name}")
                                else:
                                    st.error(f"Failed to index {uploaded_file.name}")
                            else:
                                st.error(f"Failed to extract content from {uploaded_file.name}")
                        
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        
                        finally:
                            # Clean up temp file
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                            
                            # Add separator between files
                            if i < len(uploaded_files) - 1:
                                st.markdown("---")
                    
                    # Final status
                    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
                    st.info(f"Processing complete. Vector database contains {collection_info.vectors_count} vectors.")
        else:
            # Show placeholder when no files uploaded
            st.markdown("""
            <div class="upload-area">
                <h3>Drag and drop PDF files here</h3>
                <p>or click the "Browse files" button above</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 2: Query Documents
    with tab2:
        st.markdown('<h2>Query Documents</h2>', unsafe_allow_html=True)
        
        # Display query workflow
        display_query_workflow()
        
        # Query input area
        query = st.text_area("Enter your question:", height=100, 
                            placeholder="What would you like to know from your documents?")
        
        # Options row
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            page_filter = st.number_input("Filter by page number (optional)", 
                                        min_value=0, value=0, 
                                        help="Enter a page number to restrict search to that page, or leave at 0 to search all pages")
        
        with col2:
            top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
        
        with col3:
            show_details = st.checkbox("Show result details", value=True,
                                     help="Display metadata and relevance scores with results")
        
        # Execute query
        if st.button("Submit Query", key="query_btn"):
            if query:
                with st.spinner("Processing query..."):
                    # Convert page filter value
                    actual_page = None if page_filter == 0 else page_filter
                    
                    # Execute search
                    results = query_documents(
                        query=query,
                        embedding_models=embedding_models,
                        qdrant_client=qdrant_client,
                        top_k=top_k,
                        page_filter=actual_page
                    )
                    
                    # Display results
                    if results:
                        st.markdown(f"### Found {len(results)} relevant results:")
                        
                        for i, result in enumerate(results):
                            with st.expander(f"Result {i+1} - Page {result['metadata']['page']} - Score: {result['score']:.4f}", expanded=(i==0)):
                                # Display content
                                st.markdown(f"**Content:**\n{result['content']}")
                                
                                # Display metadata if requested
                                if show_details:
                                    st.markdown("**Metadata:**")
                                    metadata_df = pd.DataFrame({
                                        "Property": list(result['metadata'].keys()),
                                        "Value": list(result['metadata'].values())
                                    })
                                    st.dataframe(metadata_df, hide_index=True)
                    else:
                        st.warning("No results found. Try a different query or check if documents are indexed.")
            else:
                st.warning("Please enter a query to search.")
        
        # Example queries section
        with st.expander("Example Queries", expanded=False):
            st.markdown("""
            Here are some example queries you can try:
            - What is mentioned on page 5?
            - Summarize the introduction section
            - What tables are in the document?
            - Find information about [specific topic]
            - Compare information between page 3 and page 7
            """)
    
    # Tab 3: Document Management
    with tab3:
        st.markdown('<h2>Manage Documents</h2>', unsafe_allow_html=True)
        
        # Refresh document list button
        if st.button("Refresh Document List", key="refresh_docs"):
            st.experimental_rerun()
        
        # Get list of PDFs
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
        
        if pdf_files:
            st.markdown(f"### {len(pdf_files)} Documents Indexed")
            
            # Show documents in a clean format
            for pdf in pdf_files:
                pdf_path = os.path.join(PDF_FOLDER, pdf)
                file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Size in MB
                
                # Try to get page count
                try:
                    with fitz.open(pdf_path) as doc:
                        page_count = len(doc)
                except:
                    page_count = "Unknown"
                
                # Document card
                with st.expander(pdf, expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**File:** {pdf}")
                        st.markdown(f"**Size:** {file_size:.2f} MB")
                        st.markdown(f"**Pages:** {page_count}")
                        st.markdown(f"**Path:** {pdf_path}")
                    
                    with col2:
                        # Document actions
                        if st.button("Delete", key=f"del_{pdf}"):
                            try:
                                # Remove file
                                os.remove(pdf_path)
                                st.success(f"Deleted {pdf}")
                                # Note: In a full implementation, we would also remove the vectors
                                # from Qdrant that correspond to this document
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error deleting file: {str(e)}")
            
            # Database management options
            st.markdown("### Database Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Reset Vector Database", key="reset_db"):
                    try:
                        # Delete and recreate collection
                        qdrant_client.delete_collection(COLLECTION_NAME)
                        qdrant_client.create_collection(
                            collection_name=COLLECTION_NAME,
                            vectors_config={
                                "all-MiniLM-L6-v2": VectorParams(size=384, distance=Distance.COSINE),
                                "bm25": VectorParams(size=65536, distance=Distance.JACCARD),
                                "colbertv2.0": VectorParams(size=128, distance=Distance.COSINE),
                            }
                        )
                        st.success("Vector database reset successfully")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error resetting database: {str(e)}")
            
            with col2:
                if st.button("Reindex All Documents", key="reindex_all"):
                    try:
                        # Reset collection
                        qdrant_client.delete_collection(COLLECTION_NAME)
                        qdrant_client.create_collection(
                            collection_name=COLLECTION_NAME,
                            vectors_config={
                                "all-MiniLM-L6-v2": VectorParams(size=384, distance=Distance.COSINE),
                                "bm25": VectorParams(size=65536, distance=Distance.JACCARD),
                                "colbertv2.0": VectorParams(size=128, distance=Distance.COSINE),
                            }
                        )
                        
                        # Process each PDF
                        with st.expander("Reindexing Status", expanded=True):
                            for i, pdf in enumerate(pdf_files):
                                pdf_path = os.path.join(PDF_FOLDER, pdf)
                                
                                st.markdown(f"### Reindexing: {pdf} ({i+1}/{len(pdf_files)})")
                                
                                # Process PDF
                                documents = process_pdf(pdf_path)
                                
                                if documents:
                                    # Create chunks
                                    chunks = create_chunks(
                                        documents, 
                                        chunk_size=chunk_size, 
                                        chunk_overlap=chunk_overlap
                                    )
                                    
                                    # Index to Qdrant
                                    success = index_documents(chunks, embedding_models, qdrant_client)
                                    
                                    if success:
                                        st.success(f"Successfully reindexed {pdf}")
                                    else:
                                        st.error(f"Failed to reindex {pdf}")
                                else:
                                    st.error(f"Failed to extract content from {pdf}")
                                
                                # Add separator between files
                                if i < len(pdf_files) - 1:
                                    st.markdown("---")
                        
                        st.success("All documents reindexed successfully")
                    except Exception as e:
                        st.error(f"Error reindexing documents: {str(e)}")
        else:
            st.info("No documents have been indexed yet. Go to the 'Upload & Process' tab to add documents.")
            
            # Option to create sample data
            if st.button("Create Sample Data", key="create_sample"):
                try:
                    # Generate a simple PDF with sample text
                    from reportlab.pdfgen import canvas
                    from reportlab.lib.pagesizes import letter
                    
                    sample_path = os.path.join(PDF_FOLDER, "sample_document.pdf")
                    c = canvas.Canvas(sample_path, pagesize=letter)
                    
                    # Add some text
                    c.setFont("Helvetica", 12)
                    c.drawString(100, 750, "Sample PDF Document")
                    c.drawString(100, 730, "This is a sample document for testing the RAG system.")
                    c.drawString(100, 710, "It contains multiple pages with different content.")
                    
                    # Add a simple table
                    c.drawString(100, 670, "Sample Table:")
                    c.drawString(100, 650, "ID | Name | Value")
                    c.drawString(100, 630, "1 | Item A | 100")
                    c.drawString(100, 610, "2 | Item B | 200")
                    c.drawString(100, 590, "3 | Item C | 300")
                    
                    # Add more pages
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    c.drawString(100, 750, "Page 2 - More Information")
                    c.drawString(100, 730, "This is the second page of the sample document.")
                    c.drawString(100, 710, "It contains different information for testing.")
                    
                    # Save the PDF
                    c.save()
                    
                    st.success("Sample document created successfully. Refresh to see it in the list.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error creating sample data: {str(e)}")

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 10px; border-top: 1px solid #eee;">
        <p style="color: #666; font-size: 0.8rem;">
            PDF RAG System with Qdrant | Built with Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
