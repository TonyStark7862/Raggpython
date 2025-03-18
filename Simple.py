import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
import hashlib
import os
from pathlib import Path
import io

# Create directories
VECTORDB_DIR = Path("./vectordb")
VECTORDB_DIR.mkdir(exist_ok=True, parents=True)

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
        st.session_state.retriever = None
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Process PDF file
if uploaded_file is not None:
    # Calculate file hash for caching
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    cache_file = VECTORDB_DIR / f"{file_hash}.pkl"
    
    # Check if we need to process the file
    if st.session_state.retriever is None:
        with st.spinner("Processing PDF..."):
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
                    
                    st.success("PDF processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

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
