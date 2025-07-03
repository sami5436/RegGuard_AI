import streamlit as st
from source_code.app.app import display_ui
from source_code.core.graph import RAGSystem

# --- Configuration ---
VECTOR_STORE_PATH = "vector_store"
DOCUMENTS_PATH = "documents"

def get_rag_system():
    """Initializes and returns the RAGSystem instance from session state."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = RAGSystem(
            docs_path=DOCUMENTS_PATH, 
            vector_store_path=VECTOR_STORE_PATH
        )
    return st.session_state.rag_system

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="RegGuard AI", page_icon="compliance.png", layout="wide")

    # Initialize the RAG system
    rag_system = get_rag_system()

    # Display the user interface
    display_ui(rag_system, VECTOR_STORE_PATH)

if __name__ == "__main__":
    main()
