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

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
    
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
    """Create a new collection if it doesn't exist."""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            st.success(f"✅ Collection '{collection_name}' created")
        else:
            st.info(f"Using existing collection '{collection_name}'")
    except Exception as e:
        st.error(f"Error creating collection: {e}")

# Process PDF and add to Qdrant
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
        
        # Prepare points for upload
        points = [
            models.PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={"text": chunk, "chunk_id": idx}
            )
            for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks))
        ]
        
        # Upload to collection
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        st.success(f"✅ Added {len(points)} chunks to collection")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        raise e

# Search for relevant chunks in Qdrant
def search_chunks(collection_name, query, limit=4):
    """Search for chunks similar to the query."""
    try:
        # Get model and client
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        
        # Generate embedding for query
        query_embedding = model.encode([query])[0]
        
        # Search in collection
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
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
                    # Retrieve relevant chunks
                    sources = search_chunks(st.session_state.collection_name, prompt)
                    
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
