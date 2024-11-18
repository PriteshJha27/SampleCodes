from typing import Annotated, Dict, List, Tuple, TypedDict
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_core.messages import BaseMessage, HumanMessage
import operator
from langgraph.graph import END, Graph
from langgraph.prebuilt import ToolExecutor
import json
import os

# Define state types
class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_step: str
    documents: List[Document]
    chunks: List[Document]
    vectorstore: FAISS
    retrieved_docs: List[Document]
    final_answer: str
    query: str
    error: str

# Initialize tools
def create_rag_tools():
    """Create tools for the RAG pipeline"""
    
    def load_pdf(input_dict: Dict) -> Dict:
        """Load PDF and return documents"""
        try:
            pdf_path = input_dict.get("pdf_path")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            return {
                "documents": documents,
                "status": "success"
            }
        except Exception as e:
            return {
                "documents": [],
                "status": "error",
                "error": str(e)
            }

    def chunk_documents(input_dict: Dict) -> Dict:
        """Split documents into chunks"""
        try:
            documents = input_dict.get("documents", [])
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            return {
                "chunks": chunks,
                "status": "success"
            }
        except Exception as e:
            return {
                "chunks": [],
                "status": "error",
                "error": str(e)
            }

    def create_embeddings(input_dict: Dict) -> Dict:
        """Create embeddings and vectorstore"""
        try:
            chunks = input_dict.get("chunks", [])
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            return {
                "vectorstore": vectorstore,
                "status": "success"
            }
        except Exception as e:
            return {
                "vectorstore": None,
                "status": "error",
                "error": str(e)
            }

    def retrieve_docs(input_dict: Dict) -> Dict:
        """Retrieve relevant documents"""
        try:
            vectorstore = input_dict.get("vectorstore")
            query = input_dict.get("query")
            docs = vectorstore.similarity_search(query, k=3)
            return {
                "retrieved_docs": docs,
                "status": "success"
            }
        except Exception as e:
            return {
                "retrieved_docs": [],
                "status": "error",
                "error": str(e)
            }

    def query_llm(input_dict: Dict) -> Dict:
        """Get answer from LLM"""
        try:
            vectorstore = input_dict.get("vectorstore")
            query = input_dict.get("query")
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            answer = qa_chain.run(query)
            return {
                "final_answer": answer,
                "status": "success"
            }
        except Exception as e:
            return {
                "final_answer": "",
                "status": "error",
                "error": str(e)
            }

    # Create tool dictionary
    tools = {
        "load_pdf": load_pdf,
        "chunk_documents": chunk_documents,
        "create_embeddings": create_embeddings,
        "retrieve_docs": retrieve_docs,
        "query_llm": query_llm
    }
    
    return tools

# Define state updates
def update_state(state: AgentState, tool_output: Dict) -> AgentState:
    """Update state based on tool output"""
    # Copy state to avoid mutations
    new_state = state.copy()
    
    # Update state based on tool output
    if tool_output.get("status") == "success":
        if "documents" in tool_output:
            new_state["documents"] = tool_output["documents"]
        if "chunks" in tool_output:
            new_state["chunks"] = tool_output["chunks"]
        if "vectorstore" in tool_output:
            new_state["vectorstore"] = tool_output["vectorstore"]
        if "retrieved_docs" in tool_output:
            new_state["retrieved_docs"] = tool_output["retrieved_docs"]
        if "final_answer" in tool_output:
            new_state["final_answer"] = tool_output["final_answer"]
    else:
        new_state["error"] = tool_output.get("error", "Unknown error")
        
    return new_state

# Define workflow nodes
def create_workflow_nodes(tools: Dict):
    """Create workflow nodes for the graph"""
    
    tool_executor = ToolExecutor(tools)
    
    def load_pdf_node(state: AgentState) -> Tuple[AgentState, str]:
        """Node for loading PDF"""
        result = tool_executor.execute(
            "load_pdf",
            {"pdf_path": state.get("pdf_path", "")}
        )
        new_state = update_state(state, result)
        
        if new_state.get("error"):
            return new_state, "error"
        return new_state, "chunk"

    def chunk_node(state: AgentState) -> Tuple[AgentState, str]:
        """Node for chunking documents"""
        result = tool_executor.execute(
            "chunk_documents",
            {"documents": state.get("documents", [])}
        )
        new_state = update_state(state, result)
        
        if new_state.get("error"):
            return new_state, "error"
        return new_state, "embed"

    def embed_node(state: AgentState) -> Tuple[AgentState, str]:
        """Node for creating embeddings"""
        result = tool_executor.execute(
            "create_embeddings",
            {"chunks": state.get("chunks", [])}
        )
        new_state = update_state(state, result)
        
        if new_state.get("error"):
            return new_state, "error"
        return new_state, "retrieve"

    def retrieve_node(state: AgentState) -> Tuple[AgentState, str]:
        """Node for retrieving documents"""
        result = tool_executor.execute(
            "retrieve_docs",
            {
                "vectorstore": state.get("vectorstore"),
                "query": state.get("query", "")
            }
        )
        new_state = update_state(state, result)
        
        if new_state.get("error"):
            return new_state, "error"
        return new_state, "answer"

    def answer_node(state: AgentState) -> Tuple[AgentState, str]:
        """Node for getting LLM answer"""
        result = tool_executor.execute(
            "query_llm",
            {
                "vectorstore": state.get("vectorstore"),
                "query": state.get("query", "")
            }
        )
        new_state = update_state(state, result)
        
        if new_state.get("error"):
            return new_state, "error"
        return new_state, "end"

    def error_node(state: AgentState) -> Tuple[AgentState, str]:
        """Node for handling errors"""
        print(f"Error encountered: {state.get('error', 'Unknown error')}")
        return state, "end"

    return {
        "load": load_pdf_node,
        "chunk": chunk_node,
        "embed": embed_node,
        "retrieve": retrieve_node,
        "answer": answer_node,
        "error": error_node
    }

class RAGGraph:
    def __init__(self):
        # Create tools
        self.tools = create_rag_tools()
        
        # Create workflow nodes
        self.nodes = create_workflow_nodes(self.tools)
        
        # Create the graph
        self.workflow = Graph()
        
        # Add nodes
        self.workflow.add_node("load", self.nodes["load"])
        self.workflow.add_node("chunk", self.nodes["chunk"])
        self.workflow.add_node("embed", self.nodes["embed"])
        self.workflow.add_node("retrieve", self.nodes["retrieve"])
        self.workflow.add_node("answer", self.nodes["answer"])
        self.workflow.add_node("error", self.nodes["error"])
        
        # Add edges
        self.workflow.add_edge("load", "chunk")
        self.workflow.add_edge("chunk", "embed")
        self.workflow.add_edge("embed", "retrieve")
        self.workflow.add_edge("retrieve", "answer")
        self.workflow.add_edge("answer", END)
        self.workflow.add_edge("error", END)
        
        # Set entry point
        self.workflow.set_entry_point("load")
        
        # Compile the graph
        self.app = self.workflow.compile()
    
    def process_query(self, pdf_path: str, query: str) -> str:
        """Process a query through the RAG pipeline"""
        # Initialize state
        initial_state = AgentState(
            messages=[],
            current_step="load",
            documents=[],
            chunks=[],
            vectorstore=None,
            retrieved_docs=[],
            final_answer="",
            query=query,
            error="",
            pdf_path=pdf_path
        )
        
        # Run the workflow
        final_state = self.app.invoke(initial_state)
        
        # Return result or error
        if final_state.get("error"):
            return f"Error: {final_state['error']}"
        return final_state.get("final_answer", "No answer generated")

# Example usage
if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    # Initialize RAG graph
    rag = RAGGraph()
    
    # Process a query
    pdf_path = "path/to/your/document.pdf"
    query = "What is the main topic of the document?"
    
    # Get response
    response = rag.process_query(pdf_path, query)
    print(f"Response: {response}")
