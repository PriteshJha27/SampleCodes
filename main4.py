# D:/Pritesh/VS Code Workspace/Preparations/Streamlit/3.png


import streamlit as st

# Define page navigation
def navigate_to(page):
    st.session_state.page = page

# Homepage layout with improved alignment
def homepage():
    st.title("Gen AI Recipes")

    # Styling for custom layout
    st.markdown("""
        <style>
            .container {
                display: flex;
                align-items: center;
                margin-bottom: 20px;
            }
            .image-container {
                flex: 0 0 100px;
                text-align: center;
            }
            .content-container {
                flex: 1;
                padding-left: 20px;
            }
            .button-description {
                color: #666;
                font-size: 14px;
            }
        </style>
    """, unsafe_allow_html=True)

    # 1. LLM Call
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.image("D:/Pritesh/VS Code Workspace/Preparations/Streamlit/1.png", caption="Meta Llama", width=80, use_column_width=False, output_format="PNG")
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.subheader("1. LLM Call")
    if st.button("LLM Call", key="llm_call"):
        navigate_to("LLM Call")
    st.markdown('<p class="button-description">An LLM call to Llama model</p>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # 2. RAG
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.image("D:/Pritesh/VS Code Workspace/Preparations/Streamlit/2.png", caption="RAG", width=80, use_column_width=False, output_format="PNG")
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.subheader("2. RAG")
    if st.button("RAG", key="rag"):
        navigate_to("RAG")
    st.markdown('<p class="button-description">A Simple RAG Application</p>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # 3. Specialized RAG
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.image("D:/Pritesh/VS Code Workspace/Preparations/Streamlit/3.png", caption="Semantic Reranker", width=80, use_column_width=False, output_format="PNG")
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.subheader("3. Specialized RAG")
    if st.button("Specialized RAG", key="specialized_rag"):
        navigate_to("Specialized RAG")
    st.markdown('<p class="button-description">Specialized and more accurate version of RAG using Re-ranking</p>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # 4. Agentic RAG
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.image("D:/Pritesh/VS Code Workspace/Preparations/Streamlit/4.png", caption="LangGraph", width=80, use_column_width=False, output_format="PNG")
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.subheader("4. Agentic RAG")
    if st.button("Agentic RAG", key="agentic_rag"):
        navigate_to("Agentic RAG")
    st.markdown('<p class="button-description">RAG application being run using Multi-Agents by Langgraph</p>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

# Page for Llama Call
def llama_call():
    st.title("LLM Call - Llama")
    st.write("An LLM call to the Llama model.")
    user_input = st.text_input("Enter your message:")
    
    if st.button("Submit"):
        # Simulate Llama model call here
        response = f"Llama response to: {user_input}"
        st.write("### Chat")
        st.write(f"**User:** {user_input}")
        st.write(f"**Llama:** {response}")

# Page for RAG
def rag_app():
    st.title("RAG - Retrieval Augmented Generation")
    st.write("Use case for basic RAG implementation.")
    # Use the previously provided code for the RAG application logic

# Page for Specialized RAG
def specialized_rag_app():
    st.title("Specialized RAG - Retrieval with Re-ranking")
    st.write("Use case for specialized RAG using Re-ranking.")
    # Use similar RAG logic with added processing for specialized needs

# Page for Agentic RAG
def agentic_rag_app():
    st.title("Agentic RAG - Multi-Agent RAG Application")
    st.write("Use case for RAG using multi-agent approach with Langgraph.")
    # Use similar RAG logic with multi-agent handling

# Main navigation logic
if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    homepage()
elif st.session_state.page == "LLM Call":
    llama_call()
elif st.session_state.page == "RAG":
    rag_app()
elif st.session_state.page == "Specialized RAG":
    specialized_rag_app()
elif st.session_state.page == "Agentic RAG":
    agentic_rag_app()
