import streamlit as st
import asyncio
from rag import EmbeddingsWrapper, retrieval  # Custom RAG functions
from langchain.vectorstores import FAISS
import hashlib

# Function to calculate the hash of uploaded documents
def calculate_file_hash(file):
    file.seek(0)
    data = file.read()
    file.seek(0)  # Reset the file pointer after reading
    return hashlib.md5(data).hexdigest()

# Cache function to create embeddings and vectorstore
@st.cache_resource(show_spinner=False)
def create_vectorstore(documents, folder_path, model_path, config_path):
    embeddings_wrapper = EmbeddingsWrapper(folder_path=folder_path, model_path=model_path, config_path=config_path)
    vectorstore = FAISS.from_documents(documents, embeddings_wrapper)
    return embeddings_wrapper, vectorstore

# Homepage layout with improved alignment
def homepage():
    st.title("Gen AI Recipes - Q&A App")
    
    # Input configuration
    folder_path = st.text_input("Folder Path for Embeddings")
    model_path = st.text_input("Model Path for Embeddings")
    config_path = st.text_input("Config Path for Embeddings")

    # Upload a PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Calculate the hash of the uploaded file
        file_hash = calculate_file_hash(uploaded_file)

        # Save and load the document
        temp_file_path = "uploaded_document.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        uploaded_file.seek(0)

        # Load document
        loader = PyMuPDFLoader(temp_file_path)
        documents = loader.load()
        st.write(f"Loaded {len(documents)} document sections.")

        # Store documents in session state
        st.session_state['documents'] = documents

        # Generate embeddings and create vectorstore
        with st.spinner("Generating embeddings and creating a vectorstore..."):
            embeddings_wrapper, vectorstore = create_vectorstore(documents, folder_path, model_path, config_path)
            st.session_state['embeddings_wrapper'] = embeddings_wrapper
            st.session_state['vectorstore'] = vectorstore
            st.write("Vectorstore created.")

        # Navigate to RAG
        st.session_state.page = "RAG"

# RAG page
def rag():
    st.title("Q&A on your documents")

    # Retrieve documents, embeddings_wrapper, and vectorstore from session state
    documents = st.session_state.get('documents', None)
    embeddings_wrapper = st.session_state.get('embeddings_wrapper', None)
    vectorstore = st.session_state.get('vectorstore', None)

    if documents and embeddings_wrapper and vectorstore:
        query = st.text_input("Enter your query here")
        
        if st.button("Submit Query") and query:
            with st.spinner("Retrieving relevant documents..."):
                retrieved_docs = asyncio.run(retrieval(embeddings_wrapper, vectorstore, query))
                st.write(f"Found {len(retrieved_docs)} relevant documents.")
                for i, doc in enumerate(retrieved_docs, 1):
                    st.write(f"**Document {i}:** {doc[:200]}...")
    else:
        st.write("Please go back to the homepage and upload a document first.")

# Navigation setup
if 'page' not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    homepage()
elif st.session_state.page == "RAG":
    rag()
