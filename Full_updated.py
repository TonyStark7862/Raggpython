from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
import hashlib
from pathlib import Path
import os
import streamlit as st
from streamlit_chat import message
import io
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path("./cache")
VECTORDB_DIR = Path("./vectordb")
MODEL_DIR = Path("./models")

# Create directories if they don't exist
for directory in [CACHE_DIR, VECTORDB_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

async def main():
    # Function to calculate file hash for caching
    def get_file_hash(file_bytes):
        return hashlib.md5(file_bytes).hexdigest()

    async def storeDocEmbeds(file_bytes, file_hash):
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            
            # Store page numbers with text for later reference
            all_texts = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # Store both text and metadata (page number)
                    all_texts.append((text, {"page": i+1}))
            
            # Extract just the text for splitting
            corpus = [text for text, _ in all_texts]
            metadata = [meta for _, meta in all_texts]
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            # Split with metadata tracking
            chunks = []
            chunk_metadatas = []
            for i, (text, meta) in enumerate(zip(corpus, metadata)):
                split_texts = splitter.split_text(text)
                # Create metadata for each chunk from the same page
                chunk_meta = [{"page": meta["page"], "chunk": j} for j in range(len(split_texts))]
                chunks.extend(split_texts)
                chunk_metadatas.extend(chunk_meta)
            
            # Use local embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=str(MODEL_DIR / "sentence-transformers/all-mpnet-base-v2"),
                cache_folder=str(MODEL_DIR)
            )
            
            # Create vectors with metadata
            vectors = FAISS.from_texts(chunks, embeddings, metadatas=chunk_metadatas)
            
            # Save to local vectordb directory with the file hash as identifier
            vector_file = VECTORDB_DIR / f"{file_hash}.pkl"
            with open(vector_file, "wb") as f:
                pickle.dump(vectors, f)
                
            logger.info(f"Stored document embeddings at {vector_file}")
            return vectors
        except Exception as e:
            logger.error(f"Error storing document embeddings: {e}")
            raise
        
    async def getDocEmbeds(file_bytes, filename):
        # Calculate file hash for caching
        file_hash = get_file_hash(file_bytes)
        cache_file = VECTORDB_DIR / f"{file_hash}.pkl"
        
        # Check if we have this file cached
        if cache_file.exists():
            logger.info(f"Loading cached embeddings for {filename}")
            try:
                with open(cache_file, "rb") as f:
                    vectors = pickle.load(f)
                return vectors
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}. Regenerating...")
                # If loading fails, regenerate
        
        # If not cached or cache failed, generate new embeddings
        logger.info(f"Generating new embeddings for {filename}")
        return await storeDocEmbeds(file_bytes, file_hash)

    async def conversational_chat(query):
        try:
            # Get temperature from session state or use default
            temperature = st.session_state.get('temperature', 0.7)
            
            # Start response timer
            import time
            start_time = time.time()
            
            # Get retriever from session state
            retriever = st.session_state.get('retriever')
            if not retriever:
                return "Error: Document retriever not initialized. Please reload the document.", [], 0
            
            # Use the retriever to get relevant documents
            docs = retriever.get_relevant_documents(query)
            
            # Extract source information and text
            sources = []
            for i, doc in enumerate(docs):
                source_info = {
                    "page": doc.metadata.get("page", "Unknown"),
                    "chunk": doc.metadata.get("chunk", i),
                    "text": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "full_text": doc.page_content,
                    "relevance_score": "High" if i < 2 else "Medium" if i < 4 else "Low"
                }
                sources.append(source_info)
            
            # Format chat history for context (limit to last 5 exchanges for efficiency)
            recent_history = st.session_state['history'][-5:] if len(st.session_state['history']) > 5 else st.session_state['history']
            formatted_history = "\n".join([f"Human: {q}\nAI: {a}" for q, a in recent_history])
            
            # Create prompt with context and question
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Enhanced prompt with metadata and instructions
            prompt = f"""
            You are an AI assistant answering questions about a PDF document titled "{st.session_state.get('doc_name', 'document')}".
            
            Previous conversation:
            {formatted_history}
            
            Relevant document sections:
            {context}
            
            Human question: {query}
            
            Instructions:
            1. Answer based ONLY on the information in the document sections provided
            2. If the answer isn't in the provided context, say you don't know rather than making up information
            3. Provide direct, concise answers
            4. Format your response using markdown where helpful
            
            Your answer:
            """
            
            # Use abc_response directly with temperature
            answer = abc_response(prompt)
            
            # Calculate response time
            response_time = round(time.time() - start_time, 2)
            
            # Update history
            st.session_state['history'].append((query, answer))
            
            # Record metrics if not already in session state
            if 'metrics' not in st.session_state:
                st.session_state['metrics'] = {
                    'queries': 0,
                    'avg_response_time': 0,
                    'sources_used': 0
                }
            
            # Update metrics
            current_metrics = st.session_state['metrics']
            current_metrics['queries'] += 1
            current_metrics['avg_response_time'] = ((current_metrics['avg_response_time'] * (current_metrics['queries']-1)) + response_time) / current_metrics['queries']
            current_metrics['sources_used'] += len(sources)
            
            # Return both answer and sources with metrics
            return answer, sources, response_time
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
            return f"Sorry, I encountered an error: {str(e)}", [], 0

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'ready' not in st.session_state:
        st.session_state['ready'] = False

    if 'file_hash' not in st.session_state:
        st.session_state['file_hash'] = None
        
    if 'sources' not in st.session_state:
        st.session_state['sources'] = []
        
    if 'retrieval_k' not in st.session_state:
        st.session_state['retrieval_k'] = 4
        
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = 0.7
        
    if 'show_full_context' not in st.session_state:
        st.session_state['show_full_context'] = False
    
    # Function to clear chat history
    def clear_chat():
        st.session_state['history'] = []
        st.session_state['past'] = ["Hey!"]
        st.session_state['generated'] = ["👋 Hello! I've processed your document. You can now ask me any questions about it."]
        st.session_state['sources'] = []
        st.session_state['response_times'] = []
    
    # Function to reset everything 
    def reset_app():
        for key in ['history', 'past', 'generated', 'sources', 'ready', 'file_hash', 'doc_name', 'response_times', 'metrics']:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

    # Creating the chatbot interface with custom CSS
    st.set_page_config(page_title="PDF AI Assistant", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for a more professional look
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Header styling */
        .main h1 {
            color: #2C3E50;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 1.2rem 0;
            border-bottom: 2px solid #3498DB;
        }
        
        /* Card styling for messages */
        .stChatMessage {
            background-color: white;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        /* User message styling */
        [data-testid="stChatMessageUser"] {
            background-color: #E3F2FD;
        }
        
        /* Bot message styling */
        [data-testid="stChatMessageAssistant"] {
            background-color: #FFF;
        }
        
        /* Source container styling */
        .source-container {
            background-color: #f1f1f1;
            border-left: 3px solid #3498DB;
            padding: 0.8rem;
            margin: 0.5rem 0;
            border-radius: 5px;
        }
        
        /* Button styling */
        .stButton button {
            background-color: #3498DB;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            background-color: #2980B9;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-size: 0.9rem;
            color: #2C3E50;
            background-color: #EBF5FB;
            border-radius: 5px;
        }
        
        /* Status indicators */
        .status-ready {
            color: #27AE60;
            font-weight: bold;
        }
        
        .status-waiting {
            color: #E67E22;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # App header with logo
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("📚")
    with col2:
        st.title("PDF AI Assistant")

    # Sidebar for settings and file upload
    with st.sidebar:
        st.header("Document Settings")
        st.markdown("---")
        
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Model settings
            st.subheader("AI Settings")
            retrieval_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=4)
            temperature = st.slider("Response creativity", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                                   help="Higher values make output more creative, lower values make it more precise")
            
            # Display document stats if available
            if st.session_state.get('ready', False):
                st.subheader("Document Stats")
                try:
                    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
                    st.info(f"Pages: {len(reader.pages)}\nFile size: {round(len(uploaded_file.getvalue())/1024, 2)} KB")
                except:
                    pass
        
        # Additional settings
        st.markdown("---")
        st.subheader("Display Settings")
        show_full_context = st.checkbox("Always show full context", value=st.session_state.get('show_full_context', False))
        if show_full_context != st.session_state.get('show_full_context'):
            st.session_state['show_full_context'] = show_full_context
        
        # Chat management
        if st.session_state.get('ready', False):
            st.markdown("---")
            st.subheader("Chat Management")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Chat"):
                    clear_chat()
            with col2:
                if st.button("Reset App"):
                    reset_app()
        
        # About section
        st.markdown("---")
        st.markdown("### About")
        st.markdown("PDF AI Assistant helps you chat with your PDF documents using AI.")
        st.markdown("© 2025 | v1.0.0")
        
        # Show keyboard shortcuts
        with st.expander("Keyboard Shortcuts"):
            st.markdown("""
            - **Enter**: Submit question
            - **Ctrl+Z/Cmd+Z**: Undo text
            - **Escape**: Clear input field
            """)

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        current_hash = get_file_hash(file_bytes)
        
        # Only process if this is a new file or we haven't processed it yet
        if st.session_state['file_hash'] != current_hash or not st.session_state['ready']:
            # Create a progress bar for document processing
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Initialize processing
                status_text.text("Analyzing document structure...")
                progress_bar.progress(10)
                
                # Step 2: Extract text and create embeddings
                status_text.text("Extracting text from PDF...")
                progress_bar.progress(30)
                vectors = await getDocEmbeds(file_bytes, uploaded_file.name)
                
                # Step 3: Create retriever with user-defined settings
                status_text.text("Building knowledge base...")
                progress_bar.progress(70)
                retrieval_k = st.session_state.get('retrieval_k', 4)
                global retriever
                retriever = vectors.as_retriever(search_kwargs={"k": retrieval_k})
                
                # Step 4: Finalize
                status_text.text("Finalizing setup...")
                progress_bar.progress(90)
                
                # Update session state
                st.session_state['ready'] = True
                st.session_state['file_hash'] = current_hash
                st.session_state['doc_name'] = uploaded_file.name
                
                # Reset chat history when new document is loaded
                st.session_state['history'] = []
                st.session_state['sources'] = []
                if 'generated' in st.session_state:
                    st.session_state['generated'] = [f"👋 Hello! I've processed *{uploaded_file.name}*. You can now ask me any questions about this document."]
                if 'past' in st.session_state:
                    st.session_state['past'] = ["Hello"]
                
                # Complete progress
                progress_bar.progress(100)
                status_text.empty()
                
                # Show success message with document stats
                reader = PdfReader(io.BytesIO(file_bytes))
                st.success(f"✅ **{uploaded_file.name}** processed successfully! ({len(reader.pages)} pages)")
                
                logger.info(f"Successfully processed {uploaded_file.name}")
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error processing PDF: {str(e)}")
                st.session_state['ready'] = False

    st.divider()

    if st.session_state['ready']:
        # Initialize chat interface
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Welcome! You can now ask questions about the uploaded PDF."]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey!"]

        # Container for chat history
        response_container = st.container()

        # Container for text box
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Ask a question:", placeholder="e.g: Summarize the document in a few sentences", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                with st.status("Searching document and generating response...") as status:
                    try:
                        # Update status with steps
                        status.update(label="📄 Searching for relevant information...", state="running")
                        
                        # Get response with sources and timing
                        output, sources, response_time = await conversational_chat(user_input)
                        
                        # Update status
                        status.update(label="🧠 Generating answer...", state="running")
                        
                        # Add to session state
                        st.session_state['past'].append(user_input)
                        st.session_state['generated'].append(output)
                        st.session_state['sources'].append(sources)
                        
                        # Add response time to session
                        if 'response_times' not in st.session_state:
                            st.session_state['response_times'] = []
                        st.session_state['response_times'].append(response_time)
                        
                        # Complete the status
                        status.update(label=f"✅ Response generated in {response_time}s", state="complete")
                    except Exception as e:
                        logger.error(f"Error generating response: {e}")
                        status.update(label=f"❌ Error: {str(e)}", state="error")
                        st.error(f"Error generating response: {str(e)}")

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
                    
                    # Show response time if available
                    if i > 0 and i-1 < len(st.session_state.get('response_times', [])):
                        response_time = st.session_state['response_times'][i-1]
                        st.caption(f"Response time: {response_time}s")
                    
                    # Show sources for this response if available
                    if i > 0 and i-1 < len(st.session_state['sources']) and st.session_state['sources'][i-1]:
                        show_full = st.session_state.get('show_full_context', False)
                        
                        with st.expander(f"📚 View sources from PDF ({len(st.session_state['sources'][i-1])} references)"):
                            # Create tabs for different ways to view sources
                            source_tabs = st.tabs(["By Relevance", "By Page Number", "Full Context"])
                            
                            with source_tabs[0]:
                                # Sort by relevance (already sorted)
                                for j, source in enumerate(st.session_state['sources'][i-1]):
                                    with st.container():
                                        col1, col2 = st.columns([1, 4])
                                        with col1:
                                            relevance = source['relevance_score']
                                            relevance_color = "green" if relevance == "High" else "orange" if relevance == "Medium" else "gray"
                                            st.markdown(f"<span style='color:{relevance_color};font-weight:bold;'>●</span> <span style='color:{relevance_color};'>{relevance}</span>", unsafe_allow_html=True)
                                            st.markdown(f"**Page {source['page']}**")
                                        
                                        with col2:
                                            st.markdown(f"*Excerpt:* {source['text']}")
                                            if show_full:
                                                st.text_area("Full context:", source['full_text'], height=100, disabled=True)
                                            else:
                                                with st.expander("View full context"):
                                                    st.markdown(source['full_text'])
                                    st.divider()
                            
                            with source_tabs[1]:
                                # Sort by page number
                                page_sorted = sorted(st.session_state['sources'][i-1], key=lambda x: x['page'])
                                for source in page_sorted:
                                    st.markdown(f"**Page {source['page']}**")
                                    st.markdown(f"*Excerpt:* {source['text']}")
                                    if show_full:
                                        st.text_area("Full context:", source['full_text'], height=100, disabled=True)
                                    else:
                                        with st.expander("View full context"):
                                            st.markdown(source['full_text'])
                                    st.divider()
                            
                            with source_tabs[2]:
                                # Show full context for all sources
                                full_text = "\n\n".join([f"**Page {s['page']}**\n{s['full_text']}" for s in st.session_state['sources'][i-1]])
                                st.markdown(full_text)
        # Add export chat functionality
        if len(st.session_state['generated']) > 1:
            st.divider()
            export_col1, export_col2 = st.columns([3, 1])
            with export_col1:
                st.markdown("### Chat Summary")
                # Show simple chat metrics
                if 'metrics' in st.session_state:
                    metrics = st.session_state['metrics']
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Questions", metrics['queries'])
                    m2.metric("Avg. Response Time", f"{metrics['avg_response_time']:.2f}s")
                    m3.metric("Sources Referenced", metrics['sources_used'])
            
            with export_col2:
                # Create export options
                export_format = st.selectbox("Export format", ["Text", "Markdown", "HTML"])
                
                if st.button("Export Chat"):
                    # Generate chat export
                    export_content = ""
                    if export_format == "Text":
                        for q, a in zip(st.session_state['past'], st.session_state['generated']):
                            export_content += f"Q: {q}\n\nA: {a}\n\n----------\n\n"
                    elif export_format == "Markdown":
                        export_content = "# Chat with PDF: " + st.session_state.get('doc_name', 'Document') + "\n\n"
                        for q, a in zip(st.session_state['past'], st.session_state['generated']):
                            export_content += f"## Question\n\n{q}\n\n## Answer\n\n{a}\n\n---\n\n"
                    else:  # HTML
                        export_content = "<html><head><style>body{font-family:Arial;max-width:800px;margin:0 auto;padding:20px}h1{color:#2C3E50}h2{color:#3498DB}.question{background-color:#E3F2FD;padding:15px;border-radius:5px;margin:10px 0}.answer{background-color:#F8F9FA;padding:15px;border-radius:5px;margin:10px 0}</style></head><body>"
                        export_content += f"<h1>Chat with PDF: {st.session_state.get('doc_name', 'Document')}</h1>"
                        for q, a in zip(st.session_state['past'], st.session_state['generated']):
                            export_content += f"<h2>Question</h2><div class='question'>{q}</div><h2>Answer</h2><div class='answer'>{a}</div><hr>"
                        export_content += "</body></html>"
                    
                    # Create a download button
                    file_ext = ".txt" if export_format == "Text" else ".md" if export_format == "Markdown" else ".html"
                    file_name = f"chat_export{file_ext}"
                    st.download_button(
                        label="Download",
                        data=export_content,
                        file_name=file_name,
                        mime="text/plain" if export_format == "Text" else "text/markdown" if export_format == "Markdown" else "text/html"
                    )

    else:
        # Welcome screen
        if uploaded_file is None:
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <h1 style="color: #3498DB;">Welcome to PDF AI Assistant</h1>
                <img src="https://i.ibb.co/1ZsVs2x/pdf-logo.png" style="width: 150px; height: auto; margin: 2rem;">
                <h3>Upload a PDF document to get started</h3>
                <p>Ask questions about your document and get AI-powered answers with source references.</p>
                <p style="color: #7f8c8d; margin-top: 3rem;">Upload your document using the sidebar on the left.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sample questions
            st.markdown("### Sample questions you can ask:")
            sample_questions = [
                "What is the main topic of this document?",
                "Can you summarize the key points?",
                "What are the conclusions or recommendations?",
                "Explain the methodology described in the document.",
                "What evidence supports the main argument?"
            ]
            
            for question in sample_questions:
                st.markdown(f"- *{question}*")

if __name__ == "__main__":
    asyncio.run(main())
