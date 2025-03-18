import streamlit as st
import numpy as np
import os
import time
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
import PyPDF2
import io
import uuid

# Configuration for local paths - CHANGE THESE TO YOUR PREFERRED LOCATIONS
LOCAL_QDRANT_PATH = "./qdrant_qna_data"
LOCAL_MODEL_PATH = "./models/all-MiniLM-L6-v2"

# Ensure directories exist
os.makedirs(LOCAL_QDRANT_PATH, exist_ok=True)
os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

@st.cache_resource
def load_sentence_transformer():
    """Initialize the embedding model from local path or download if not present."""
    try:
        st.info(f"Loading embedding model from {LOCAL_MODEL_PATH}...")
        model = SentenceTransformer(LOCAL_MODEL_PATH)
        st.success("✅ Model loaded from local path")
    except Exception as e:
        st.warning(f"Model not found locally or error loading: {e}. Downloading model (this may take a moment)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Save the model for future use
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        model.save(LOCAL_MODEL_PATH)
        st.success("✅ Model downloaded and saved locally")
    return model

@st.cache_resource
def setup_qdrant_client():
    """Setup Qdrant client with local persistence."""
    try:
        client = QdrantClient(path=LOCAL_QDRANT_PATH)
        st.success("✅ Connected to local Qdrant database")
        return client
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {e}")
        return None

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
            st.info(f"Collection '{collection_name}' already exists")
    except Exception as e:
        st.error(f"Error creating collection: {e}")

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks for better context preservation."""
    chunks = []
    if not text:
        return chunks
    
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # Adjust end to avoid cutting words in the middle
        if end < text_length:
            # Look for a space or newline to end the chunk
            while end > start and text[end] not in [' ', '\n']:
                end -= 1
            if end == start:  # If no space found, use the original end
                end = min(start + chunk_size, text_length)
        
        # Create a chunk with metadata
        chunk = {
            "id": str(uuid.uuid4()),
            "content": text[start:end].strip()
        }
        
        if chunk["content"]:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start < 0:
            start = 0
            
    return chunks

def add_chunks_to_collection(client, collection_name, model, chunks, source_info):
    """Add text chunks to the collection."""
    try:
        if not chunks:
            st.warning("No content extracted from the document.")
            return
        
        # Extract chunk contents
        contents = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings
        with st.spinner(f"Generating embeddings for {len(chunks)} text chunks..."):
            embeddings = model.encode(contents)
        
        # Get collection info to check point count
        collection_info = client.get_collection(collection_name)
        existing_count = collection_info.points_count
        
        # Generate IDs starting after existing points
        starting_id = existing_count
        
        # Prepare points for upload
        points = [
            models.PointStruct(
                id=starting_id + idx,
                vector=embedding.tolist(),
                payload={
                    "content": chunk["content"],
                    "source": source_info,
                    "chunk_id": chunk["id"]
                }
            )
            for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks))
        ]
        
        # Upload to collection
        with st.spinner(f"Adding {len(points)} text chunks to collection..."):
            client.upsert(
                collection_name=collection_name,
                points=points
            )
        
        st.success(f"✅ Added {len(points)} text chunks to collection")
        return len(points)
    except Exception as e:
        st.error(f"Error adding chunks: {e}")
        return 0

def search_chunks(client, collection_name, model, query_text, limit=3):
    """Search for text chunks similar to the query."""
    try:
        # Generate embedding for query
        query_embedding = model.encode([query_text])[0]
        
        # Search in collection
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        
        return search_results
    except Exception as e:
        st.error(f"Error searching chunks: {e}")
        return []

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'pdfs_processed' not in st.session_state:
        st.session_state.pdfs_processed = False
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

def display_chat_history():
    """Display the chat history in a conversational format."""
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.chat_message('user').write(message['content'])
        else:
            st.chat_message('assistant').write(message['content'])

def main():
    st.set_page_config(
        page_title="PDF Chat App",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Chat with Your PDFs")
    st.markdown("""
    Upload PDF files and ask questions about their content. The system will find the most relevant information.
    
    ### Features:
    - Upload multiple PDF files
    - Chat-based interface with conversation history
    - Semantic search finds information based on meaning, not just keywords
    - Persistent storage for both the database and embedding model
    """)
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize model and client
    model = load_sentence_transformer()
    client = setup_qdrant_client()
    
    if not client:
        st.error("Failed to initialize Qdrant client. Please check the configuration.")
        return
    
    # Setup collection
    collection_name = f"pdf_chunks_{st.session_state.session_id}"
    create_collection(
        client, 
        collection_name,
        vector_size=model.get_sentence_embedding_dimension()
    )
    
    # Sidebar for settings and file upload
    with st.sidebar:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF Files", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        # Process files button
        if uploaded_files and not st.session_state.pdfs_processed:
            process_button = st.button("Process Documents", type="primary")
            if process_button:
                total_chunks = 0
                with st.spinner("Processing PDF files..."):
                    for file in uploaded_files:
                        st.info(f"Processing {file.name}...")
                        # Extract text from PDF
                        pdf_text = extract_text_from_pdf(file)
                        
                        # Split text into chunks
                        chunks = chunk_text(pdf_text)
                        
                        # Add chunks to collection
                        added_chunks = add_chunks_to_collection(
                            client, 
                            collection_name, 
                            model, 
                            chunks, 
                            file.name
                        )
                        total_chunks += added_chunks
                
                st.session_state.pdfs_processed = True
                st.session_state.uploaded_files = [file.name for file in uploaded_files]
                st.success(f"✅ Processed {len(uploaded_files)} files with {total_chunks} total chunks")
        
        st.divider()
        
        # Settings
        st.subheader("Settings")
        num_results = st.slider("Number of results to show", 1, 5, 3)
        
        st.divider()
        
        # Display collection info
        st.subheader("Session Info")
        try:
            collection_info = client.get_collection(collection_name)
            st.write(f"Session ID: {st.session_state.session_id[:8]}...")
            st.write(f"Number of Text Chunks: {collection_info.points_count}")
            
            if st.session_state.uploaded_files:
                st.write("Processed Files:")
                for file_name in st.session_state.uploaded_files:
                    st.write(f"- {file_name}")
        except Exception as e:
            st.error(f"Error fetching collection info: {e}")
            
        # Reset session button
        if st.button("Start New Session"):
            # Delete the collection
            try:
                client.delete_collection(collection_name)
            except:
                pass
            # Reset session state
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()
    
    # Main area - Chat interface
    st.divider()
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if st.session_state.pdfs_processed:
        user_input = st.chat_input("Ask a question about your documents...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            
            # Display the latest user message (this is needed to show the message immediately)
            st.chat_message('user').write(user_input)
            
            # Get relevant chunks
            with st.spinner("Searching for relevant information..."):
                results = search_chunks(client, collection_name, model, user_input, limit=num_results)
            
            # Generate response based on search results
            assistant_message = st.chat_message('assistant')
            
            if results:
                # Construct response from search results
                response = "Here's what I found in your documents:\n\n"
                
                for i, result in enumerate(results):
                    source = result.payload.get("source", "Unknown document")
                    content = result.payload.get("content", "")
                    score = result.score
                    
                    response += f"**From {source}** (Relevance: {score:.2f}):\n"
                    response += f"{content}\n\n"
                
            else:
                response = "I couldn't find specific information about that in your documents. Could you rephrase your question or ask about something else?"
            
            # Write the response
            assistant_message.write(response)
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
    else:
        st.info("Please upload and process PDF documents to start chatting.")
        
        # Sample questions section
        st.subheader("How to use this app:")
        st.markdown("""
        1. **Upload your PDF files** using the sidebar on the left
        2. Click **Process Documents** to extract and index the content
        3. Start asking questions about your documents
        4. The system will find the most relevant passages and display them
        5. Your chat history will be maintained during the session
        """)

if __name__ == "__main__":
    main()
