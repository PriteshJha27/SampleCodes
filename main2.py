import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import hashlib

# Function to calculate the hash of uploaded documents
def calculate_file_hash(file):
    file.seek(0)
    data = file.read()
    file.seek(0)  # Reset the file pointer after reading
    return hashlib.md5(data).hexdigest()

# Cache function to store embeddings and vectorstore
@st.cache_resource(show_spinner=False)
def create_vectorstore(_documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(_documents, embeddings)
    return vectorstore

# Streamlit app setup
st.title("RAG-based Document Query System")
st.write("Upload a PDF document and input your query to retrieve relevant information.")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Calculate the hash of the uploaded file
    file_hash = calculate_file_hash(uploaded_file)
    
    # Step 2: Save and load the document
    progress_bar = st.progress(0)
    with st.spinner("Loading the document..."):
        # Save the uploaded file to disk
        temp_file_path = "uploaded_document.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        uploaded_file.seek(0)  # Reset file pointer

        # Check if the file is not empty
        if uploaded_file.size == 0:
            st.error("The uploaded file is empty. Please upload a valid PDF.")
            st.stop()  # Stop execution if the file is empty

        # Load the document
        loader = PyMuPDFLoader(temp_file_path)
        documents = loader.load()
        st.write(f"Loaded {len(documents)} document sections.")
    progress_bar.progress(20)  # Update progress

    # Step 3: Generate embeddings and create vectorstore (cached)
    with st.spinner("Generating embeddings and creating a vectorstore..."):
        vectorstore = create_vectorstore(documents)
    st.write("Vectorstore created.")
    progress_bar.progress(50)  # Update progress

    # Step 4: Input query
    query = st.text_input("Enter your query here")

    if st.button("Submit Query") and query:
        # Step 5: Retrieval
        with st.spinner("Retrieving relevant documents..."):
            retrieved_docs = vectorstore.similarity_search(query)
            st.write(f"Found {len(retrieved_docs)} relevant documents.")
        progress_bar.progress(70)  # Update progress

        # Display retrieved documents
        st.write("Relevant Documents:")
        for i, doc in enumerate(retrieved_docs, 1):
            st.write(f"**Document {i}:** {doc.page_content[:200]}...")  # Show a snippet of the content

        # Step 6: LLM Call
        with st.spinner("Generating response..."):
            llm = ChatOpenAI(model="gpt-4o-mini")
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert retrieval summarizer that finds the right answer and summarizes in less than 25 words."),
                ("human", "{query}")
            ])
            chain = prompt | llm | StrOutputParser()

            # Use the retrieved documents as input for the query
            response = chain.invoke({"query": retrieved_docs})

        # Display the LLM response
        st.write("### Response Summary:")
        st.write(response)
        progress_bar.progress(100)  # Complete progress
else:
    st.write("Please upload a PDF file to proceed.")
