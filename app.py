import streamlit as st
import os
from rag_core import RAGSystem

# --- Configuration ---
VECTOR_STORE_PATH = "vector_store"
DOCUMENTS_PATH = "documents"

# --- Helper Functions ---
def get_rag_system():
    """Initializes and returns the RAGSystem instance."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = RAGSystem(
            docs_path=DOCUMENTS_PATH, 
            vector_store_path=VECTOR_STORE_PATH
        )
    return st.session_state.rag_system

def display_chat_history():
    """Displays the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Streamlit UI ---
st.set_page_config(page_title="RegGuard AI", page_icon="compliance.png", layout="wide")

# App title and description
st.title("RegGuard AI 🤖")
st.markdown("""
Welcome to RegGuard AI, an interactive RAG platform for navigating oil & gas emissions compliance. 
Ask questions about your regulatory documents and get answers backed by sources.
""")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Setup & Controls")
    st.markdown("""
    Use this section to manage your regulatory documents and the vector database that powers the search.
    """)

    # Button to build the vector store
    if st.button("Build Vector Store", help="Process documents in the 'documents' folder and create a searchable vector store. This may take a few minutes."):
        rag_system = get_rag_system()
        with st.spinner("Processing documents and building vector store... Please wait."):
            try:
                rag_system.build_vector_store()
                st.success("Vector store built successfully!")
                st.info(f"Vector store is ready at: {os.path.abspath(VECTOR_STORE_PATH)}")
            except Exception as e:
                st.error(f"An error occurred while building the vector store: {e}")

    st.markdown("---")
    st.header("About")
    st.info("""
    - **Backend**: Python, LangGraph, Ollama
    - **Vector Search**: FAISS
    - **Frontend**: Streamlit
    """)

# --- Main Chat Interface ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
display_chat_history()

# Chat input for user's question
if prompt := st.chat_input("Ask a question about your compliance documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            rag_system = get_rag_system()
            if not os.path.exists(VECTOR_STORE_PATH) or not os.listdir(VECTOR_STORE_PATH):
                 st.error("The vector store has not been built yet. Please click 'Build Vector Store' in the sidebar.", icon="⚠️")
            else:
                with st.spinner("Thinking..."):
                    result = rag_system.query(prompt)
                    
                    # Display the final answer
                    st.markdown(result.get("generation", "Sorry, I couldn't generate an answer."))
                    
                    # Display the source documents
                    with st.expander("View Sources"):
                        if "documents" in result and result["documents"]:
                            for i, doc in enumerate(result["documents"]):
                                st.markdown(f"**Source {i+1}: `{doc.metadata.get('source', 'Unknown')}`**")
                                st.markdown(doc.page_content)
                                st.markdown("---")
                        else:
                            st.warning("No source documents were found for this answer.")
                
                # Add the complete response to session state (for history)
                # We construct a string that includes the answer and a summary of sources
                source_summary = "\n\n**Sources:**\n"
                if "documents" in result and result["documents"]:
                    for doc in result["documents"]:
                        source_summary += f"- `{doc.metadata.get('source', 'N/A')}`\n"
                else:
                    source_summary = "\n\n(No sources found)"
                
                final_content = result.get("generation", "") + source_summary
                st.session_state.messages.append({"role": "assistant", "content": final_content})

        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

