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
        st.info("Currently using: Qdrant with Hybrid Retrieval")
        # Add Qdrant hybrid retrieval settings
        st.subheader("Qdrant Search Settings")
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
    """Create a new collection if it doesn't exist with text search capabilities."""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            # Create collection with optimized configuration for hybrid search
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
            st.success(f"✅ Collection '{collection_name}' created with hybrid search capabilities")
        else:
            st.info(f"Using existing collection '{collection_name}'")
    except Exception as e:
        st.error(f"Error creating collection: {e}")

# Process PDF and add to Qdrant with text payload for hybrid search
def process_pdf_qdrant(file_bytes, collection_name):
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
        
        # Get model and client
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        
        # Create embeddings for chunks
        embeddings = model.encode(chunks)
        
        # Check if collection exists and has points
        collection_info = qdrant_client.get_collection(collection_name)
        existing_count = collection_info.points_count
        
        # Skip if chunks are already added
        if existing_count > 0:
            st.info(f"Document chunks already added to collection (found {existing_count} points)")
            return
        
        # Prepare points for upload - include text field for text search in hybrid retrieval
        points = [
            models.PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    "text": chunk,  # For retrieval
                    "chunk_id": idx,
                    "content": chunk  # This field will be indexed for text search
                }
            )
            for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks))
        ]
        
        # Upload to collection
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        # Configure payload indexing for text search (needed for hybrid search)
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
        
        st.success(f"✅ Added {len(points)} chunks to collection with text indexing for hybrid search")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        raise e

# Search for relevant chunks in Qdrant using hybrid search
def search_chunks(collection_name, query, limit=4):
    """Search for chunks similar to the query using hybrid retrieval."""
    try:
        # Get model and client
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        
        # Generate embedding for query
        query_embedding = model.encode([query])[0]
        
        # Set up hybrid search parameters
        # This combines vector search with keyword-based search
        hybrid_query = models.HybridSearchParams(
            query_text=query,  # For text-based search part
            query_vector=query_embedding.tolist(),  # For semantic similarity search
            weights=(0.3, 0.7),  # Weight for (text, vector) - adjust as needed
        )
        
        # Perform hybrid search in collection
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=hybrid_query,
            limit=limit
        )
        
        return [result.payload["text"] for result in search_results]
    except Exception as e:
        st.error(f"Error searching chunks: {e}")
        return []

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
                
                if qdrant_client:
                    # Create collection with appropriate vector size
                    vector_size = model.get_sentence_embedding_dimension()
                    create_collection(qdrant_client, collection_name, vector_size)
                    
                    # Process PDF and add to collection
                    process_pdf_qdrant(file_bytes, collection_name)
                    
                    # Update session state
                    st.session_state.collection_name = collection_name
                    st.success("PDF processed successfully with Qdrant!")
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
                    # Retrieve relevant chunks using hybrid search with user-defined settings
                    if 'hybrid_weight' in st.session_state:
                        vector_weight = st.session_state.get('hybrid_weight', 0.7)
                        text_weight = 1.0 - vector_weight
                        limit = st.session_state.get('result_count', 4)
                    else:
                        vector_weight = 0.7
                        text_weight = 0.3
                        limit = 4
                        
                    # Get model and client
                    model = load_sentence_transformer()
                    qdrant_client = setup_qdrant_client()
                    
                    # Generate embedding for query
                    query_embedding = model.encode([prompt])[0]
                    
                    # Set up hybrid search parameters with user-defined weights
                    hybrid_query = models.HybridSearchParams(
                        query_text=prompt,  # For text-based search part
                        query_vector=query_embedding.tolist(),  # For semantic similarity search
                        weights=(text_weight, vector_weight),  # Weight for (text, vector)
                    )
                    
                    # Perform hybrid search
                    search_results = qdrant_client.search(
                        collection_name=st.session_state.collection_name,
                        query_vector=hybrid_query,
                        limit=limit
                    )
                    
                    sources = [result.payload["text"] for result in search_results]
                    
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
