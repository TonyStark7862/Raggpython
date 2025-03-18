import os
import uuid
import tempfile
import streamlit as st
import pandas as pd
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HybridRAG-UI")

# Import your LLM response function (replace with your actual import)
try:
    from your_llm_module import abc_response
except ImportError:
    # Mock implementation for testing
    def abc_response(prompt):
        return f"This is a mock response. In production, replace with your actual LLM function. Prompt length: {len(prompt)} characters."

# Import the RAG system
from back import HybridRAG

# Set page configuration
st.set_page_config(
    page_title="Advanced Hybrid RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional UI
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: #0f2e4c;
    }
    .sub-header {
        font-size: 1.6rem;
        font-weight: 600;
        margin: 1.2rem 0 0.8rem 0;
        color: #2c3e50;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        color: #34495e;
    }
    
    /* Chat Messages */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        line-height: 1.5;
    }
    .user-message {
        background-color: #e6f7ff;
        border-left: 4px solid #1890ff;
    }
    .bot-message {
        background-color: #f9f9f9;
        border-left: 4px solid #52c41a;
    }
    
    /* Sources Section */
    .source-container {
        margin-top: 0.5rem;
        padding: 0.8rem;
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 0.3rem;
        font-size: 0.85rem;
    }
    .source-header {
        font-weight: 600;
        color: #333;
        margin-bottom: 0.3rem;
    }
    .source-meta {
        color: #555;
        margin-bottom: 0.3rem;
        font-size: 0.8rem;
    }
    .source-content {
        border-left: 2px solid #ddd;
        padding-left: 0.8rem;
        margin-top: 0.3rem;
        color: #333;
    }
    .method-tag {
        display: inline-block;
        padding: 0.1rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 0.3rem;
    }
    .method-intersection {
        background-color: #f6ffed;
        color: #52c41a;
        border: 1px solid #b7eb8f;
    }
    .method-faiss {
        background-color: #e6f7ff;
        color: #1890ff;
        border: 1px solid #91d5ff;
    }
    .method-bm25 {
        background-color: #fff7e6;
        color: #fa8c16;
        border: 1px solid #ffd591;
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }
    .loader {
        border: 5px solid #f3f3f3;
        border-top: 5px solid #3498db;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    .status-success {
        background-color: #52c41a;
    }
    .status-warning {
        background-color: #faad14;
    }
    .status-error {
        background-color: #f5222d;
    }
    
    /* Document Cards */
    .doc-card {
        padding: 0.8rem;
        border: 1px solid #e0e0e0;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
        background-color: #f9f9f9;
    }
    .doc-card-header {
        font-weight: 600;
        color: #333;
        margin-bottom: 0.3rem;
    }
    
    /* Input Area */
    .query-input-container {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "rag_system" not in st.session_state:
        # Initialize with local paths
        data_dir = Path("./rag_data")
        st.session_state.rag_system = HybridRAG(data_dir=str(data_dir))
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "sources" not in st.session_state:
        st.session_state.sources = st.session_state.rag_system.get_document_sources()
    
    if "system_stats" not in st.session_state:
        st.session_state.system_stats = st.session_state.rag_system.get_stats()
    
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "hybrid_weight": 0.7,
            "top_k": 5,
            "show_sources": True,
        }

# Initialize session
init_session_state()

# Sidebar for document upload and settings
with st.sidebar:
    st.markdown("<h2 class='sub-header'>📁 Document Management</h2>", unsafe_allow_html=True)
    
    # File uploader for PDFs and TXT
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload PDF or text documents for analysis"
    )
    
    # Process uploaded files
    if uploaded_files and st.button("Process Documents", type="primary"):
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)
        
        # Save uploaded files to temp directory
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        try:
            # Save files
            for file in uploaded_files:
                status_text.text(f"Saving {file.name}...")
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                file_paths.append(file_path)
            
            # Ingest documents with progress updates
            ingest_result = st.session_state.rag_system.ingest_documents(
                file_paths, 
                progress_callback=progress_callback
            )
            
            # Update sources and stats
            st.session_state.sources = st.session_state.rag_system.get_document_sources()
            st.session_state.system_stats = st.session_state.rag_system.get_stats()
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.text("✅ Documents processed successfully!")
            
            # Display success message
            st.success(ingest_result)
            
        except Exception as e:
            # Handle errors
            logger.error(f"Error processing documents: {e}")
            progress_bar.progress(1.0)
            status_text.text("❌ Error processing documents")
            st.error(f"Error: {str(e)}")
            
        finally:
            # Clean up temp files
            try:
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                os.rmdir(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning temp files: {e}")
    
    # Display system statistics
    with st.expander("System Statistics", expanded=False):
        stats = st.session_state.system_stats
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Qdrant Points", stats.get("qdrant_points", 0))
            st.metric("FAISS Vectors", stats.get("faiss_vectors", 0))
        with col2:
            st.metric("BM25 Documents", stats.get("bm25_documents", 0))
            st.metric("Active Sessions", stats.get("sessions", 0))
        
        if "sources" in stats and stats["sources"]:
            st.markdown("### Indexed Documents")
            for source in stats["sources"]:
                st.markdown(f"- {source}")
        else:
            st.info("No documents indexed yet.")
    
    # Settings
    st.markdown("<h2 class='sub-header'>⚙️ Settings</h2>", unsafe_allow_html=True)
    
    # Session management
    if st.button("Clear Chat History", help="Clear the current conversation history"):
        st.session_state.chat_history = []
        st.session_state.rag_system.clear_session(st.session_state.session_id)
        st.success("Chat history cleared!")
    
    # Retrieval settings
    with st.expander("Retrieval Settings", expanded=False):
        st.session_state.settings["hybrid_weight"] = st.slider(
            "Dense-Sparse Balance",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.settings["hybrid_weight"],
            step=0.1,
            help="Balance between dense vectors (1.0) and sparse vectors (0.0)"
        )
        
        st.session_state.settings["top_k"] = st.slider(
            "Results Count",
            min_value=3,
            max_value=15,
            value=st.session_state.settings["top_k"],
            step=1,
            help="Number of results to retrieve"
        )
        
        st.session_state.settings["show_sources"] = st.checkbox(
            "Show Sources by Default",
            value=st.session_state.settings["show_sources"],
            help="Automatically expand source information for answers"
        )

# Main content area
col1, col2 = st.columns([2, 8])
with col1:
    st.image("https://img.icons8.com/fluent/96/000000/document-exchange.png", width=80)
with col2:
    st.markdown("<h1 class='main-header'>Advanced Hybrid RAG System</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="margin-bottom: 1.5rem;">
        This system uses a sophisticated hybrid retrieval approach with:
        <ul style="margin-bottom: 0.5rem;">
            <li>Dense semantic search (FAISS + all-mpnet-base-v2)</li>
            <li>Sparse lexical search (BM25)</li>
            <li>Optimized intersection-first retrieval</li>
            <li>Document structure awareness</li>
        </ul>
    </div>
    """, 
    unsafe_allow_html=True
)

# Info message if no documents
if not st.session_state.sources:
    st.info("👋 Start by uploading some documents in the sidebar to index them for question answering.")

# Display chat interface
st.markdown("<h2 class='sub-header'>💬 Chat Interface</h2>", unsafe_allow_html=True)

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(
            f'<div class="chat-message user-message">'
            f'<strong>You:</strong> {message["content"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-message bot-message">'
            f'<strong>Assistant:</strong> {message["content"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("View Sources", expanded=st.session_state.settings["show_sources"]):
                for i, source in enumerate(message["sources"]):
                    # Determine method tag class
                    method = source.get("source", "hybrid")
                    method_class = f"method-{method}" if method in ["intersection", "faiss", "bm25"] else ""
                    
                    st.markdown(
                        f'<div class="source-container">'
                        f'<div class="source-header">Source {i+1}: {source["metadata"].get("source", "Unknown")}</div>'
                        f'<div class="source-meta">'
                        f'<span>Page {source["metadata"].get("page", "N/A")} | '
                        f'{source["metadata"].get("section_type", "content").capitalize()} | '
                        f'Score: {source["score"]:.2f}</span>'
                        f'<span class="method-tag {method_class}">{method.upper()}</span>'
                        f'</div>'
                        f'<div class="source-content">{source["text"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
st.markdown('</div>', unsafe_allow_html=True)

# Chat input
st.markdown('<div class="query-input-container">', unsafe_allow_html=True)
user_input = st.text_input(
    "Ask a question about your documents:",
    key="user_input",
    placeholder="What is this document about? What are the key findings?"
)
st.markdown('</div>', unsafe_allow_html=True)

# Process user input
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
    })
    
    # Create a placeholder for the assistant's response
    response_placeholder = st.empty()
    response_placeholder.markdown(
        '<div class="loading-container">'
        '<div class="loader"></div>'
        '<p>Generating response...</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    
    try:
        # Generate answer using the RAG system
        start_time = time.time()
        answer, sources = st.session_state.rag_system.generate_answer(
            query=user_input,
            session_id=st.session_state.session_id,
            abc_response_func=abc_response,
            top_k=st.session_state.settings["top_k"],
        )
        query_time = time.time() - start_time
        
        # Add assistant message to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "query_time": query_time,
        })
        
        # Remove the placeholder
        response_placeholder.empty()
        
        # Rerun to update the chat display
        st.rerun()
        
    except Exception as e:
        # Handle errors
        logger.error(f"Error generating response: {e}")
        response_placeholder.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span>Advanced Hybrid RAG System | Production-Ready Implementation</span>
        <span style="color: #888; font-size: 0.8rem;">All document processing is performed locally</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    # App is already running
    pass
