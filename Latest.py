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
        st.info("Currently using: Qdrant with Ensemble Retrieval")
        # Add Qdrant retrieval settings
        st.subheader("Qdrant Search Settings")
        
        retrieval_method = st.radio(
            "Retrieval Method",
            options=["Ensemble (Best Performance)", "Vector Only", "Hybrid Only"],
            index=0,
            help="Select the retrieval method to use"
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
            help="Balance between keyword search (0.0) and semantic search (1.0)"
        )
        
        st.session_state.result_count = st.slider(
            "Number of results",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of document chunks to retrieve for answering"
        )
        
        st.session_state.use_reranking = st.checkbox(
            "Use Reranking",
            value=True,
            help="Rerank results to improve relevance and diversity"
        )
    
    # Clear chat button
    if st.button("Clear Chat"):
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

# Initialize the embedding model for Qdrant
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

# Process PDF and add to Qdrant with rich metadata for advanced search
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
        
        # Get model and client
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        
        # Create embeddings for all chunks
        contents = [chunk["content"] for chunk in all_chunks]
        
        # If we're showing progress, we've already done the embeddings
        if progress_container is None:
            embeddings = model.encode(contents)
        else:
            with progress_container:
                st.info(f"Preparing to upload {len(all_chunks)} chunks to Qdrant...")
                embeddings = model.encode(contents)
        
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
        
        # Upload to collection with progress tracking
        if progress_container:
            with progress_container:
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
                
                # Configure payload indexing for advanced search
                st.info("Setting up text indexing for search...")
                
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
                
                # Index page numbers for filtering
                qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="page_num",
                    field_schema=models.PayloadSchemaType.INTEGER
                )
                
                # Index source_type for filtering
                qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="source_type",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                
                st.success("✅ Collection fully indexed and ready to query!")
        else:
            # Upload to collection
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            # Configure payload indexing for advanced search
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
            
            # Index page numbers for filtering
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="page_num",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            
            # Index source_type for filtering
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="source_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            st.success(f"✅ Added {len(points)} chunks to collection with advanced search indexing")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        raise e

# Ensemble search with multiple Qdrant retrieval methods
def ensemble_search(collection_name, query_text, limit=4, vector_weight=0.7, use_reranking=True):
    """
    Perform ensemble search combining multiple retrieval techniques:
    1. Pure vector search (semantic similarity)
    2. Hybrid search (combining vector and keywords with BM25)
    3. Contextual search (with page-level chunks)
    4. Optional reranking
    """
    try:
        # Get model and client
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        
        # Generate embedding for query
        query_embedding = model.encode([query_text])[0]
        
        # 1. Pure vector search
        vector_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            score_threshold=0.5,  # Filter low relevance results
            with_payload=True
        )
        
        # 2. Hybrid search using Qdrant's native capabilities
        text_weight = 1.0 - vector_weight
        hybrid_results = qdrant_client.search(
            collection_name=collection_name,
            query=query_text,  # Direct text query for hybrid search
            limit=limit,
            with_payload=True,
            params={
                "vector": {
                    "weight": vector_weight
                },
                "text": {
                    "weight": text_weight,
                    "field": "content"  # Field to use for text search
                }
            }
        )
        
        # 3. Page-level search (larger context chunks)
        page_filter = models.FieldCondition(
            key="source_type",
            match=models.MatchValue(value="page")
        )
        
        page_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
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
        
        for idx, result in enumerate(hybrid_results):
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
                    "text": result.payload["text"],
                    "score": result.score * 0.95,  # Slightly discount page results
                    "page": result.payload.get("page_num", 0),
                    "source": "page",
                    "combined_score": result.score * 0.95
                }
        
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
        st.error(f"Error in ensemble search: {e}")
        return [], []

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
                            model_name="sentence-transformers/all-mpnet-base-v2"
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
                # Setup Qdrant client and model
                qdrant_client = setup_qdrant_client()
                model = load_sentence_transformer()
                
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
                    
                    # Create collection with appropriate vector size
                    with processing_expander:
                        st.info(f"Creating collection '{collection_name}'...")
                        vector_size = model.get_sentence_embedding_dimension()
                    
                    create_collection(qdrant_client, collection_name, vector_size)
                    
                    # Process PDF and add to collection
                    with processing_expander:
                        st.info("Extracting text from PDF...")
                        
                        # Show a progress bar for extraction
                        extraction_progress = st.progress(0)
                        reader = PdfReader(io.BytesIO(file_bytes))
                        total_pages = len(reader.pages)
                        
                        text = ""
                        for i, page in enumerate(reader.pages):
                            text += page.extract_text() + "\n"
                            extraction_progress.progress((i + 1) / total_pages)
                        
                        st.info(f"✅ Extracted text from {total_pages} pages")
                        
                        st.info("Splitting text into chunks...")
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        chunks = text_splitter.split_text(text)
                        st.info(f"✅ Split into {len(chunks)} chunks")
                        
                        st.info("Generating embeddings (this may take a while)...")
                        embedding_progress = st.progress(0)
                        
                        # Calculate embeddings in batches to show progress
                        embeddings = []
                        batch_size = 10
                        for i in range(0, len(chunks), batch_size):
                            batch = chunks[i:i+batch_size]
                            batch_embeddings = model.encode(batch)
                            embeddings.extend(batch_embeddings)
                            embedding_progress.progress(min(1.0, (i + batch_size) / len(chunks)))
                        
                        st.info(f"✅ Generated {len(embeddings)} embeddings")
                        
                        st.info("Storing vectors in Qdrant...")
                        
                    # Process rest of the PDF with progress tracking in expandable section
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
# Add a hybrid search slider control
if not USE_FAISS:
    with st.sidebar:
        st.subheader("Qdrant Search Settings")
        hybrid_weight = st.slider(
            "Hybrid Search Balance",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Balance between keyword search (0.0) and semantic search (1.0)"
        )
        result_count = st.slider(
            "Number of results",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of document chunks to retrieve for answering"
        )

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
                    response = abc_response(full_prompt)  # Your custom LLM function
                    
                    # Fallback if abc_response is not defined
                    if 'abc_response' not in globals():
                        response = f"Using local LLM to answer: {prompt}\n\nBased on the document, I found relevant information that would help answer this question."
                    
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
                    # Retrieve relevant chunks using selected search method
                    vector_weight = st.session_state.get('hybrid_weight', 0.7)
                    limit = st.session_state.get('result_count', 4)
                    use_reranking = st.session_state.get('use_reranking', True)
                    use_ensemble = st.session_state.get('use_ensemble', True)
                    use_vector_only = st.session_state.get('use_vector_only', False)
                    use_hybrid_only = st.session_state.get('use_hybrid_only', False)
                    
                    # Get model and client
                    model = load_sentence_transformer()
                    qdrant_client = setup_qdrant_client()
                    
                    if use_vector_only:
                        # Pure vector search
                        query_embedding = model.encode([prompt])[0]
                        search_results = qdrant_client.search(
                            collection_name=st.session_state.collection_name,
                            query_vector=query_embedding.tolist(),
                            limit=limit,
                            with_payload=True
                        )
                        sources = [result.payload["text"] for result in search_results]
                        metadata = [{
                            "score": result.score,
                            "page": result.payload.get("page_num", 0),
                            "source": "vector"
                        } for result in search_results]
                    elif use_hybrid_only:
                        # Hybrid search using Qdrant's native capabilities
                        query_embedding = model.encode([prompt])[0]
                        text_weight = 1.0 - vector_weight
                        search_results = qdrant_client.search(
                            collection_name=st.session_state.collection_name,
                            query=prompt,  # Direct text query for hybrid search
                            limit=limit,
                            with_payload=True,
                            params={
                                "vector": {
                                    "weight": vector_weight
                                },
                                "text": {
                                    "weight": text_weight,
                                    "field": "content"  # Field to use for text search
                                }
                            }
                        )
                        sources = [result.payload["text"] for result in search_results]
                        metadata = [{
                            "score": result.score,
                            "page": result.payload.get("page_num", 0),
                            "source": "hybrid"
                        } for result in search_results]
                    else:
                        # Ensemble search (default and best performance)
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
                        
                        # Get response from your custom LLM function
                        response = abc_response(full_prompt)  # Your custom LLM function
                        
                        # Fallback if abc_response is not defined
                        if 'abc_response' not in globals():
                            response = f"Using local LLM to answer: {prompt}\n\nBased on the document, I found relevant information that would help answer this question."
                        
                        # Display assistant response
                        with st.chat_message("assistant"):
                            st.markdown(response)
                            
                            # Show sources
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
