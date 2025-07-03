import streamlit as st
import os

def display_sidebar(rag_system, vector_store_path):
    """Displays the sidebar with controls."""
    with st.sidebar:
        st.header("Setup & Controls")
        st.markdown("Manage your regulatory documents and the vector database.")

        if st.button("Build Vector Store"):
            with st.spinner("Processing documents and building vector store..."):
                try:
                    rag_system.build_vector_store()
                    st.success("Vector store built successfully!")
                    st.info(f"Ready at: {os.path.abspath(vector_store_path)}")
                except Exception as e:
                    st.error(f"Error building vector store: {e}")
        
        st.markdown("---")
        st.header("About")
        st.info("""
        - **Backend**: Python, LangGraph, Ollama
        - **Vector Search**: FAISS
        - **Frontend**: Streamlit
        """)

def display_chat_history():
    """Displays the chat history from session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_chat_input(rag_system, vector_store_path):
    """Handles user input from the chat interface."""
    if prompt := st.chat_input("Ask a question about your compliance documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not os.path.exists(vector_store_path) or not os.listdir(vector_store_path):
                st.error("Vector store not built. Please click 'Build Vector Store' in the sidebar.", icon="⚠️")
                return

            with st.spinner("Thinking..."):
                try:
                    result = rag_system.query(prompt)
                    answer = result.get("generation", "Sorry, I couldn't generate an answer.")
                    st.markdown(answer)
                    
                    # Display sources if they exist
                    if "documents" in result and result["documents"]:
                        with st.expander("View Sources"):
                            for i, doc in enumerate(result["documents"]):
                                source = doc.metadata.get('source', 'Unknown')
                                st.markdown(f"**Source {i+1}: `{os.path.basename(source)}`**")
                                st.markdown(doc.page_content)
                                st.markdown("---")
                    
                    # Store full response in history
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

def display_ui(rag_system, vector_store_path):
    """Main function to display the entire Streamlit UI."""
    st.title("RegGuard AI 🤖")
    st.markdown("An interactive RAG platform for navigating oil & gas emissions compliance.")

    display_sidebar(rag_system, vector_store_path)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    display_chat_history()
    handle_chat_input(rag_system, vector_store_path)
