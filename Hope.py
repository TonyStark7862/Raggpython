# app.py
import os
import hashlib
import tempfile
from pathlib import Path
import streamlit as st
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF for PDF processing
from fastembed import TextEmbedding, LateInteractionTextEmbedding
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, Distance, VectorParams, SparseVector
import torch

# Configure application paths
BASE_DIR = Path("./data")
BASE_DIR.mkdir(exist_ok=True)
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / "qdrant_db"
DB_PATH.mkdir(exist_ok=True)

# Set model paths as environment variables to ensure fastembed downloads to our local directory
os.environ["FASTEMBED_CACHE_PATH"] = str(MODEL_DIR)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODEL_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

# Configure page
st.set_page_config(
    page_title="Local RAG Chat",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

def get_document_hash(file_bytes: bytes) -> str:
    """Generate a unique hash for the document content"""
    return hashlib.md5(file_bytes).hexdigest()

def load_models():
    """Load the embedding models, with progress indicators"""
    with st.spinner("Loading dense embedding model..."):
        # Load using SentenceTransformer directly for more control
        dense_embedding_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=str(MODEL_DIR))
    
    # We don't need to load anything for BM25 - it's computed on-the-fly
    
    with st.spinner("Loading late interaction model..."):
        late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    
    return dense_embedding_model, None, late_interaction_embedding_model

def preprocess_for_bm25(text: str) -> List[str]:
    """Tokenize text for BM25"""
    # Simple preprocessing - lowercase and split on whitespace
    # In a full implementation, you might want to add stopword removal, stemming, etc.
    return text.lower().split()

def extract_text_from_pdf(pdf_file) -> List[str]:
    """Extract text chunks from PDF file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    doc = fitz.open(tmp_path)
    chunks = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        
        # Simple chunking by paragraphs, can be improved
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks.extend(paragraphs)
    
    # Clean up temp file
    os.unlink(tmp_path)
    
    # Ensure chunks aren't too short - combine if needed
    processed_chunks = []
    current_chunk = ""
    
    for chunk in chunks:
        if len(current_chunk) + len(chunk) < 512:  # arbitrary chunk size
            current_chunk += " " + chunk if current_chunk else chunk
        else:
            if current_chunk:
                processed_chunks.append(current_chunk)
            current_chunk = chunk
    
    if current_chunk:  # Add the last chunk
        processed_chunks.append(current_chunk)
    
    return processed_chunks

def create_bm25_sparse_vector(bm25, query_tokens: List[str], vocabulary: Dict[str, int]) -> Tuple[List[int], List[float]]:
    """
    Create a sparse vector for Qdrant from BM25 scores.
    
    Args:
        bm25: The BM25 model
        query_tokens: Tokenized query
        vocabulary: Dictionary mapping tokens to indices
        
    Returns:
        Tuple of (indices, values) for sparse vector
    """
    # Get scores for each token in the vocabulary for this query
    scores = {}
    for token in query_tokens:
        if token in vocabulary:
            idx = vocabulary[token]
            # Get BM25 score for this token
            score = bm25.idf.get(token, 0.0)
            if score > 0:
                scores[idx] = float(score)
    
    # Convert to sorted indices and values
    indices = sorted(scores.keys())
    values = [scores[idx] for idx in indices]
    
    return indices, values

def build_bm25_model_and_vocab(text_chunks: List[str]) -> Tuple[BM25Okapi, Dict[str, int]]:
    """Build BM25 model and vocabulary from text chunks"""
    # Tokenize all chunks
    tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in text_chunks]
    
    # Build vocabulary (token -> index mapping)
    vocabulary = {}
    token_idx = 0
    for tokens in tokenized_chunks:
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = token_idx
                token_idx += 1
    
    # Create BM25 model
    bm25 = BM25Okapi(tokenized_chunks)
    
    return bm25, vocabulary

def process_document(file, models, doc_hash):
    """Process document and create embeddings"""
    dense_model, _, late_model = models
    cache_file = CACHE_DIR / f"{doc_hash}.pkl"
    
    # Check if we've processed this file before
    if cache_file.exists():
        st.success("Loading document from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Extract text chunks
    text_chunks = extract_text_from_pdf(file)
    
    if not text_chunks:
        st.error("No text could be extracted from the PDF")
        return None
    
    # Generate embeddings with progress bar
    progress_bar = st.progress(0)
    
    st.info(f"Processing {len(text_chunks)} text chunks...")
    
    # Generate dense embeddings
    with st.spinner("Generating dense embeddings..."):
        dense_embeddings = dense_model.encode(text_chunks, show_progress_bar=False)
        progress_bar.progress(0.33)
    
    # Build BM25 model and vocabulary
    with st.spinner("Building BM25 model..."):
        bm25_model, vocabulary = build_bm25_model_and_vocab(text_chunks)
        # Create sparse representations for each document
        # We'll save tokenized chunks and vocabulary to use during query time
        tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in text_chunks]
        progress_bar.progress(0.66)
    
    # Generate late interaction embeddings
    with st.spinner("Generating late interaction embeddings..."):
        late_embeddings = list(late_model.embed(chunk for chunk in text_chunks))
        progress_bar.progress(1.0)
    
    # Cache the results
    result = {
        "chunks": text_chunks,
        "tokenized_chunks": tokenized_chunks,
        "vocabulary": vocabulary, 
        "dense_embeddings": dense_embeddings,
        "bm25_model": bm25_model,
        "late_embeddings": late_embeddings
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    return result

def create_collection(client, collection_name, embeddings_data):
    """Create a collection in Qdrant with the right configuration"""
    # Check if collection already exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name in collection_names:
        st.info(f"Collection {collection_name} already exists, using it")
        return
    
    # Get sample embeddings for sizing
    dense_sample = embeddings_data["dense_embeddings"][0]
    late_sample = embeddings_data["late_embeddings"][0][0]  # First vector of first late embedding
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "all-MiniLM-L6-v2": models.VectorParams(
                size=len(dense_sample),
                distance=models.Distance.COSINE,
            ),
            "colbertv2.0": models.VectorParams(
                size=len(late_sample),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                )
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        }
    )
    
    st.success(f"Created collection: {collection_name}")

def index_document(client, collection_name, embeddings_data):
    """Index document embeddings into Qdrant"""
    chunks = embeddings_data["chunks"]
    dense_embeddings = embeddings_data["dense_embeddings"]
    tokenized_chunks = embeddings_data["tokenized_chunks"]
    vocabulary = embeddings_data["vocabulary"]
    late_embeddings = embeddings_data["late_embeddings"]
    
    # Create sparse vectors for each document using vocabulary indices
    points = []
    for idx, (dense_emb, tokens, late_emb, chunk) in enumerate(
        zip(dense_embeddings, tokenized_chunks, late_embeddings, chunks)
    ):
        # Create sparse vector for this document
        indices = []
        values = []
        
        # For each token in this document, add its index and a value of 1.0
        for token in set(tokens):  # Use set to count each token only once per doc
            if token in vocabulary:
                indices.append(vocabulary[token])
                values.append(1.0)  # Simple boolean presence
        
        # Create sparse vector
        sparse_vector = models.SparseVector(
            indices=indices,
            values=values
        )
        
        point = PointStruct(
            id=idx,
            vector={
                "all-MiniLM-L6-v2": dense_emb.tolist(),  # Convert numpy to list
                "bm25": sparse_vector,
                "colbertv2.0": late_emb,
            },
            payload={"document": chunk}
        )
        points.append(point)
    
    # Batch upsert to avoid memory issues with large documents
    batch_size = 100
    total_batches = (len(points) + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0)
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        progress = (i + len(batch)) / len(points)
        progress_bar.progress(progress)
    
    st.success(f"Indexed {len(points)} chunks into collection: {collection_name}")

def search_documents(client, collection_name, query, models, top_k=5):
    """Search for documents using hybrid search with reranking"""
    dense_model, _, late_model = models
    
    # Get the embeddings cache for the current document
    # In a real app, you'd need to know which document is active
    # Here we'll retrieve the cache file based on collection name
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    doc_cache = None
    
    for cache_file in cache_files:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            # Create a test collection name from hash to see if it matches
            file_hash = cache_file.stem
            test_name = f"doc_{file_hash[:10]}"
            if test_name == collection_name:
                doc_cache = cache_data
                break
    
    if not doc_cache:
        st.error("Document cache not found. Please reprocess the document.")
        return []
    
    # Retrieve BM25 model and vocabulary from cache
    vocabulary = doc_cache["vocabulary"]
    
    # Generate query embeddings
    dense_vector = dense_model.encode([query])[0].tolist()
    
    # Generate sparse BM25 vector for query
    query_tokens = preprocess_for_bm25(query)
    
    # Create sparse vector with indices and weights
    indices = []
    values = []
    
    # For each token in the query
    for token in query_tokens:
        if token in vocabulary:
            indices.append(vocabulary[token])
            values.append(1.0)  # Simple term presence
    
    sparse_vector = models.SparseVector(indices=indices, values=values)
    
    # Generate late interaction query vector
    late_vector = next(late_model.query_embed(query))
    
    # Set up prefetch for hybrid search
    prefetch = [
        models.Prefetch(
            query=dense_vector,
            using="all-MiniLM-L6-v2",
            limit=20,
        ),
        models.Prefetch(
            query=sparse_vector,
            using="bm25",
            limit=20,
        ),
    ]
    
    # Execute search with reranking
    results = client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=late_vector,
        using="colbertv2.0",
        with_payload=True,
        limit=top_k,
    )
    
    return results

def generate_response(query, context_chunks):
    """
    In a full implementation, this would use a language model to generate a response.
    For this example, we'll just return the most relevant chunks.
    """
    response = "Based on the documents, here's what I found:\n\n"
    for i, chunk in enumerate(context_chunks, 1):
        response += f"{i}. {chunk}\n\n"
    return response

def clear_chat_history():
    st.session_state.messages = []

def main():
    st.title("📚 Local RAG Chat System")
    
    # Sidebar for document upload and controls
    with st.sidebar:
        st.header("Document Processing")
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        
        if st.button("Clear Chat History"):
            clear_chat_history()
        
        # Display tech stack info
        st.divider()
        st.subheader("Tech Stack")
        st.markdown("""
        - **Vector DB**: Qdrant (local)
        - **Embeddings**: FastEmbed
        - **PDF Processing**: PyMuPDF
        - **Interface**: Streamlit
        """)
    
    # Main content area
    if uploaded_file:
        # Generate document hash for caching
        file_bytes = uploaded_file.getvalue()
        doc_hash = get_document_hash(file_bytes)
        collection_name = f"doc_{doc_hash[:10]}"
        st.session_state.collection_name = collection_name
        
        # Initialize Qdrant client
        client = QdrantClient(path=str(DB_PATH))
        
        # Load models (with caching via st.cache_resource)
        if "models" not in st.session_state:
            with st.spinner("Loading models..."):
                st.session_state.models = load_models()
        
        # Process document
        with st.spinner("Processing document..."):
            embeddings_data = process_document(uploaded_file, st.session_state.models, doc_hash)
            
            if embeddings_data:
                # Create collection and index document
                create_collection(client, collection_name, embeddings_data)
                index_document(client, collection_name, embeddings_data)
                
                st.success("Document processed and indexed successfully!")
    
    # Chat interface
    st.header("Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if a document has been processed
        if st.session_state.collection_name:
            # Initialize Qdrant client
            client = QdrantClient(path=str(DB_PATH))
            
            # Search for relevant context
            with st.spinner("Searching for relevant information..."):
                search_results = search_documents(
                    client, 
                    st.session_state.collection_name, 
                    prompt, 
                    st.session_state.models
                )
                
                # Extract context chunks
                context_chunks = [point.payload["document"] for point in search_results]
                
                # Generate response
                response = generate_response(prompt, context_chunks)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            # No document uploaded yet
            with st.chat_message("assistant"):
                st.markdown("Please upload a document first so I can answer questions about it.")
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Please upload a document first so I can answer questions about it."
            })

if __name__ == "__main__":
    main()
