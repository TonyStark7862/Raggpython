import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
import time
import pandas as pd
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import plotly.express as px
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Chat container */
    .chat-container {
        border-radius: 10px;
        margin-bottom: 15px;
    }
    
    /* User message styling */
    .user-message {
        background-color: #e6f3ff;
        border-left: 4px solid #0066cc;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background-color: #f0f7f0;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    /* Source container styling */
    .source-container {
        background-color: #f8f8f8;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    
    /* Source header styling */
    .source-header {
        font-weight: bold;
        margin-bottom: 5px;
        color: #555;
    }
    
    /* Relevance indicator */
    .relevance-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .relevance-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .relevance-low {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Improve sidebar appearance */
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    
    /* Custom file uploader */
    .custom-file-upload {
        border: 1px solid #ccc;
        display: inline-block;
        padding: 6px 12px;
        cursor: pointer;
        border-radius: 4px;
        background-color: #f8f9fa;
    }
    
    /* Title styling */
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 4px;
        font-weight: 500;
    }
    
    /* Chat input styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    
    /* Custom expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Import our custom RAG system
# Assuming the EnhancedBestRAG class is in a file called enhanced_bestrag.py
try:
    from enhanced_bestrag import EnhancedBestRAG
except ImportError:
    st.error("Please make sure enhanced_bestrag.py is in the same directory as this script.")
    st.stop()

# Initialize session state for chat history and RAG instance
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_instance" not in st.session_state:
    st.session_state.rag_instance = EnhancedBestRAG(collection_name="streamlit_docs")

if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = set()

if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

def evaluate_relevance(score: float) -> Dict[str, Any]:
    """
    Evaluate the relevance score and return appropriate styling and message
    
    Args:
        score: The relevance score from the search
        
    Returns:
        Dict with class name and description
    """
    if score >= 0.8:
        return {
            "class": "relevance-high",
            "icon": "✅",
            "description": "High relevance - Very reliable information",
            "color": "#28a745"
        }
    elif score >= 0.65:
        return {
            "class": "relevance-medium", 
            "icon": "✓",
            "description": "Medium relevance - Generally reliable",
            "color": "#ffc107"
        }
    else:
        return {
            "class": "relevance-low",
            "icon": "⚠️",
            "description": "Low relevance - May be tangentially related",
            "color": "#dc3545"
        }

def process_pdf(pdf_file) -> Dict[str, Any]:
    """
    Process a PDF file and add it to the RAG system
    
    Args:
        pdf_file: The uploaded PDF file
        
    Returns:
        Dict with status and message
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Write the uploaded file to the temporary file
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        # Get the filename
        file_name = pdf_file.name
        
        # Check if already processed
        if file_name in st.session_state.uploaded_pdfs:
            os.unlink(tmp_path)  # Remove temp file
            return {
                "status": "warning",
                "message": f"'{file_name}' was already processed. Skipping."
            }
            
        # Process the PDF
        metadata = {
            "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_size": len(pdf_file.getvalue())
        }
        
        # Extract text before storing embeddings to count pages
        with open(tmp_path, "rb") as pdf_file_obj:
            # First extract text to count pages
            texts = st.session_state.rag_instance._extract_pdf_text_per_page(tmp_path)
            num_pages = len(texts)
            
            # Now store embeddings
            st.session_state.rag_instance.store_pdf_embeddings(
                pdf_path=tmp_path,
                pdf_name=file_name,
                metadata=metadata
            )
        
        # Update page count
        st.session_state.total_pages += num_pages
        
        # Add to processed set
        st.session_state.uploaded_pdfs.add(file_name)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "message": f"Successfully processed '{file_name}' ({num_pages} pages)",
            "pages": num_pages
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing PDF: {str(e)}"
        }

def format_message(message: Dict[str, Any]) -> None:
    """
    Display a chat message with enhanced styling in the Streamlit UI
    
    Args:
        message: The message to display with role, content and sources
    """
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>You:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>Assistant:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Display sources if available in the message
        if "sources" in message and message["sources"]:
            with st.expander("📚 Source Documents", expanded=False):
                # Create tabs for different source views
                source_tab, visual_tab = st.tabs(["Document Sources", "Relevance Visualization"])
                
                with source_tab:
                    for i, source in enumerate(message["sources"]):
                        score = source['score']
                        relevance = evaluate_relevance(score)
                        
                        st.markdown(f"""
                        <div class="source-container">
                            <div class="source-header">
                                Source {i+1}: {source['pdf_name']} (Page {source['page_number']})
                            </div>
                            <div>
                                <span class="{relevance['class']}">
                                    {relevance['icon']} Relevance Score: {score:.2f} - {relevance['description']}
                                </span>
                            </div>
                            <details>
                                <summary>View Text Extract</summary>
                                <div style="padding: 10px; background-color: #f9f9f9; border-radius: 5px; margin-top: 5px;">
                                    {source['text'][:500]}{"..." if len(source['text']) > 500 else ""}
                                </div>
                            </details>
                        </div>
                        <hr style="margin: 15px 0;">
                        """, unsafe_allow_html=True)
                
                with visual_tab:
                    # Create visualization of relevance scores
                    source_data = []
                    for i, source in enumerate(message["sources"]):
                        source_data.append({
                            "Source": f"Source {i+1}: {source['pdf_name']} (P{source['page_number']})",
                            "Score": source['score'],
                            "RelevanceLevel": evaluate_relevance(source['score'])['description'].split('-')[0].strip()
                        })
                    
                    source_df = pd.DataFrame(source_data)
                    
                    # Create bar chart with Altair
                    chart = alt.Chart(source_df).mark_bar().encode(
                        x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y('Source:N', sort='-x'),
                        color=alt.Color('RelevanceLevel:N', 
                                      scale=alt.Scale(
                                          domain=['High relevance', 'Medium relevance', 'Low relevance'],
                                          range=['#28a745', '#ffc107', '#dc3545']
                                      )),
                        tooltip=['Source', 'Score', 'RelevanceLevel']
                    ).properties(
                        title='Document Relevance Scores',
                        height=min(len(source_data) * 50, 300)
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    st.markdown("""
                    ### Understanding Relevance Scores
                    
                    - **High relevance (0.80-1.00)**: Highly reliable information directly answering your query
                    - **Medium relevance (0.65-0.79)**: Generally reliable information related to your query
                    - **Low relevance (below 0.65)**: Information that may be only tangentially related
                    
                    The system uses multiple retrieval methods and reranking to find the most relevant content.
                    """)

def generate_rag_response(query: str, rag_instance: EnhancedBestRAG) -> str:
    """
    Generate a response using RAG and LLM
    
    Args:
        query: The user's question
        rag_instance: The RAG system instance
        
    Returns:
        str: The generated response
    """
    try:
        # Get relevant documents
        search_results = rag_instance.search(query=query, limit=5)
        
        if not search_results:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or upload more documents."
        
        # Format the context from retrieved documents
        context = ""
        for i, result in enumerate(search_results):
            text = result["payload"]["text"]
            pdf_name = result["payload"]["pdf_name"]
            page_num = result["payload"]["page_number"]
            score = result["score"]
            
            context += f"Document {i+1} (Source: {pdf_name}, Page: {page_num}, Relevance: {score:.2f}):\n{text}\n\n"
        
        # Construct prompt with retrieved context
        prompt = f"""
        CONTEXT:
        {context}
        
        QUESTION: {query}
        
        Using ONLY the information provided in the context above, please answer the question.
        If the answer is not contained in the context, state "I don't have enough information to answer that question."
        """
        
        # Use the abc_response function as instructed
        response = abc_response(prompt)
        
        return response
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Sidebar with professional styling
with st.sidebar:
    st.image("https://via.placeholder.com/150x80?text=RAG+Assistant", width=150)
    
    colored_header(label="Document Management", description="Upload and process your documents", color_name="blue-70")
    
    add_vertical_space(1)
    
    uploaded_files = st.file_uploader(
        "Upload PDF documents", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        col1, col2 = st.columns([3, 1])
        with col1:
            process_btn = st.button("Process PDFs", use_container_width=True)
        with col2:
            st.button("Clear", use_container_width=True, key="clear_upload_field")
        
        if process_btn:
            processed_count = 0
            total_pages = 0
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            for i, pdf_file in enumerate(uploaded_files):
                status_placeholder.info(f"Processing {pdf_file.name}...")
                result = process_pdf(pdf_file)
                
                if result["status"] == "success":
                    processed_count += 1
                    total_pages += result.get("pages", 0)
                    status_placeholder.success(result["message"])
                elif result["status"] == "warning":
                    status_placeholder.warning(result["message"])
                else:
                    status_placeholder.error(result["message"])
                    
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if processed_count > 0:
                st.success(f"Successfully processed {processed_count} PDFs with {total_pages} total pages")
    
    add_vertical_space(1)
    
    # Show document library
    if st.session_state.uploaded_pdfs:
        colored_header(label="Document Library", description=f"{len(st.session_state.uploaded_pdfs)} documents, {st.session_state.total_pages} pages", color_name="blue-70")
        
        # Create a container with max height and scrolling
        doc_container = st.container()
        with doc_container:
            for pdf in sorted(st.session_state.uploaded_pdfs):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%;">
                        📄 {pdf}
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("🗑️", key=f"del_{pdf}", help=f"Delete {pdf}"):
                        st.session_state.rag_instance.delete_pdf_embeddings(pdf)
                        st.session_state.uploaded_pdfs.remove(pdf)
                        st.session_state.total_pages -= 1  # This is approximate since we don't track per-file page count
                        st.rerun()
        
        # Clear all button
        st.button("Clear All Documents", use_container_width=True, key="clear_all_docs", 
                  help="Delete all documents from the system",
                  on_click=lambda: [
                      st.session_state.rag_instance.delete_pdf_embeddings(pdf) for pdf in list(st.session_state.uploaded_pdfs)
                  ] + [
                      setattr(st.session_state, 'uploaded_pdfs', set()),
                      setattr(st.session_state, 'total_pages', 0)
                  ])
    
    add_vertical_space(2)
    
    colored_header(label="Conversation Options", description="Manage your chat", color_name="blue-70")
    
    # Clear chat button
    st.button("Clear Chat History", use_container_width=True, key="clear_chat", 
              help="Erase the current conversation history",
              on_click=lambda: setattr(st.session_state, 'chat_history', []))

# Main chat interface
st.markdown("<h1 style='text-align: center;'>📚 Document Intelligence Assistant</h1>", unsafe_allow_html=True)

# Add description
st.markdown("""
<div style='text-align: center; margin-bottom: 20px;'>
Ask questions about your documents and get accurate answers with source verification
</div>
""", unsafe_allow_html=True)

# Container for chat messages
chat_container = st.container()
with chat_container:
    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            format_message(message)
    else:
        # Welcome message
        welcome_container = st.container()
        with welcome_container:
            st.markdown("""
            <div style="padding: 20px; background-color: #f8f9fa; border-radius: 10px; text-align: center; margin: 40px 0;">
                <h2>👋 Welcome to the Document Intelligence Assistant!</h2>
                <p style="font-size: 1.1em; margin: 20px 0;">
                    I can help you extract insights from your PDF documents with the power of retrieval augmented generation.
                </p>
                <div style="display: flex; justify-content: center; margin-top: 20px;">
                    <div style="text-align: left; max-width: 80%;">
                        <h4>How to use:</h4>
                        <ol>
                            <li>Upload your PDFs using the sidebar</li>
                            <li>Click 'Process PDFs' to index them</li>
                            <li>Ask questions about your documents</li>
                            <li>View source documents to verify information</li>
                        </ol>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Chat input
input_container = st.container()
with input_container:
    # Create a placeholder for potential warnings
    warning_placeholder = st.empty()
    
    # Check if documents are available
    if not st.session_state.uploaded_pdfs:
        warning_placeholder.warning("Please upload and process some PDF documents first to enable chat functionality.")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents...", disabled=len(st.session_state.uploaded_pdfs) == 0):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message (need to rerun to show it)
        st.rerun()

# Handle the response after rerun (if there's a new user message)
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    # Get the last user message
    last_user_msg = st.session_state.chat_history[-1]["content"]
    
    # Show typing indicator
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if not st.session_state.uploaded_pdfs:
                response = "Please upload and process some PDF documents first."
            else:
                # Get assistant response
                response = generate_rag_response(last_user_msg, st.session_state.rag_instance)
    
    # Get source information for display
    sources = []
    if st.session_state.uploaded_pdfs:
        with st.spinner("Retrieving sources..."):
            search_results = st.session_state.rag_instance.search(last_user_msg, limit=5)
            for result in search_results:
                sources.append({
                    "pdf_name": result["payload"]["pdf_name"],
                    "page_number": result["payload"]["page_number"],
                    "text": result["payload"]["text"],
                    "score": result["score"]
                })
    
    # Add assistant response with sources to chat history
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": response,
        "sources": sources
    })
    
    # Force a rerun to display the assistant's message
    st.rerun()

# Footer with info about the system
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; font-size: 0.8em; color: #666;">
    <p>EnhancedBestRAG Document Assistant • Retrieval-Augmented Generation with Hybrid Search</p>
</div>
""", unsafe_allow_html=True)
