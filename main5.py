import streamlit as st
import asyncio
from app.routers.rag import (EmbeddingsWrapper, retrieval, call_llama_rag, embed_query,
                             custom_similarity_search, create_vectorstore)
from langchain.vectorstores import FAISS
from PyMuPDF import fitz  # Assuming you are using PyMuPDF for PDF handling
import os

# ----------------- Home Page Function -----------------
def homepage():
    st.title("Gen AI Recipe - RAG Workflow")
    
    # Sidebar for chunk configuration and vectorstore name
    chunk_size = st.sidebar.number_input("Chunk Size", min_value=1, value=1000, step=100)
    chunk_overlap = st.sidebar.number_input("Chunk Overlap", min_value=0, value=200, step=50)
    vectorstore_name = st.sidebar.text_input("Vectorstore Name", "my_vectorstore")
    
    # PDF upload section
    uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
    
    # Progress bar
    progress_bar = st.progress(0)

    if uploaded_files:
        # Combine PDFs
        with st.spinner("Combining uploaded PDFs..."):
            merged_pdf_text = ""
            for uploaded_file in uploaded_files:
                # Read each PDF file and extract text
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                for page in doc:
                    merged_pdf_text += page.get_text()
            st.write("Merged PDFs and extracted content.")
        
        # Update progress
        progress_bar.progress(25)

        # Generate embeddings and create vectorstore
        with st.spinner("Generating embeddings and creating vectorstore..."):
            embeddings_wrapper = EmbeddingsWrapper(folder_path="your_folder_path",
                                                   model_path="your_model_path",
                                                   config_path="your_config_path")
            # Call the backend embedding function with merged text, chunk size, and overlap
            vectorstore = create_vectorstore(merged_pdf_text, embeddings_wrapper, chunk_size, chunk_overlap, vectorstore_name)
        
        # Update progress
        progress_bar.progress(75)
        st.write(f"Vectorstore '{vectorstore_name}' created successfully.")
        
        # Update to RAG page
        st.session_state['vectorstore_name'] = vectorstore_name
        st.session_state['page'] = "RAG"
        st.experimental_rerun()

# ----------------- RAG Page Function -----------------
def rag_page():
    st.title("Q&A with the Documents")

    # Sidebar configurations for RAG settings
    domain = st.sidebar.text_input("Domain", "General")
    system_role = st.sidebar.text_input("System Role", "Information Extractor")
    system_prompt = st.sidebar.text_area("System Prompt", "Please answer based on the provided documents.")
    response_mode = st.sidebar.selectbox("Select Mode", ["Llama Model", "RAG", "Reranker", "Agentic RAG"])

    # Load vectorstore if created
    vectorstore_name = st.session_state.get('vectorstore_name', None)
    if vectorstore_name:
        st.write(f"Using vectorstore: {vectorstore_name}")
        
        # Load the vectorstore
        vectorstore_path = os.path.join("vectorstores", vectorstore_name)
        vectorstore = FAISS.load_local(vectorstore_path)
    
    # Query input
    query = st.text_input("Enter your query here:")
    
    if st.button("Submit Query") and query:
        # Handle different modes
        if response_mode == "Llama Model":
            # Directly call Llama model with the query
            response = asyncio.run(call_llama_rag(system_prompt, query))
            st.write("### Llama Model Response")
            st.write(response)
        else:
            # Embed query and retrieve documents
            with st.spinner("Embedding query and retrieving documents..."):
                embed_vector = embed_query(query)
                retrieved_docs = custom_similarity_search(embed_vector, vectorstore)
                st.write("### Retrieved Documents")
                for i, doc in enumerate(retrieved_docs):
                    st.write(f"Document {i+1}: {doc[:200]}...")
            
            # Process with Llama model
            with st.spinner("Processing with Llama..."):
                # Call the Llama model with the retrieved documents
                response = asyncio.run(call_llama_rag(system_prompt, "\n".join(retrieved_docs)))
                st.write("### Llama Model Response")
                st.write(response)

# ----------------- Main Navigation -----------------
if 'page' not in st.session_state:
    st.session_state['page'] = "Home"

if st.session_state['page'] == "Home":
    homepage()
elif st.session_state['page'] == "RAG":
    rag_page()
