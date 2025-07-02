# RegGuard AI: Local RAG for Compliance Documents

RegGuard AI is an interactive, self-hosted platform designed to help you navigate complex regulatory documents, specifically tailored for oil & gas emissions compliance. It uses a powerful Retrieval-Augmented Generation (RAG) pipeline to provide answers to your questions, citing the specific sources from your documents.

The entire system runs locally on your machine, ensuring your data remains private and secure.

## Features

- **Interactive UI**: A user-friendly chat interface built with Streamlit.
- **Advanced RAG Workflow**: Powered by LangGraph to create a robust, multi-step process (retrieve, grade, generate).
- **Local First**: Uses Ollama to run powerful open-source LLMs (like Llama 3) on your own hardware. No API keys needed.
- **Efficient Search**: Employs FAISS for fast and efficient vector similarity search.
- **Source-Cited Answers**: Every answer is backed by the relevant text chunks from your source documents.
- **Metadata-Aware**: The system is designed to handle document metadata for better context and filtering (this can be extended in `rag_core.py`).

## How It Works

1.  **Document Processing**: The application reads your `.txt` files from the `documents` directory.
2.  **Chunking & Embedding**: It splits the documents into smaller, manageable chunks and uses the Ollama model to convert them into numerical representations (embeddings).
3.  **Vector Store**: These embeddings are stored in a FAISS vector index, which is saved locally in the `vector_store` directory. This is a one-time process (or whenever you update documents).
4.  **Querying**: When you ask a question:
    a. The question is converted into an embedding.
    b. FAISS finds the most relevant document chunks from the vector store.
    c. LangGraph orchestrates a workflow: it grades the relevance of the retrieved chunks.
    d. The question and the relevant, graded chunks are passed to the LLM.
    e. The LLM generates a human-readable answer based on the provided context.
5.  **Display**: The answer and the source documents are displayed in the Streamlit UI.

## Setup and Installation

Follow these steps to get RegGuard AI running on your local machine.

### Step 1: Install Ollama

You must have Ollama installed and running.

1.  Go to [https://ollama.com/](https://ollama.com/) and download the application for your operating system (macOS, Linux, Windows).
2.  After installation, open your terminal or command prompt and pull a model. We recommend `llama3`.

    ```bash
    ollama run llama3
    ```

3.  Once the model is downloaded, you can leave Ollama running in the background.

### Step 2: Project Setup

1.  **Create a project folder** and place the provided files (`app.py`, `rag_core.py`, `requirements.txt`) inside it.
2.  **Create a Python virtual environment**. This is highly recommended to avoid dependency conflicts.

    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required Python packages**:

    ```bash
    pip install -r requirements.txt
    ```

### Step 3: Add Documents

1.  Create a folder named `documents` in your project directory.
2.  Place your regulatory documents inside this folder. **They must be plain text (`.txt`) files.**
3.  Sample documents (`sample_epa_regulation.txt`, `sample_ghg_reporting_rule.txt`) have been provided for you to test the system immediately.

## How to Run the Application

1.  **Build the Vector Store**:
    -   Make sure your terminal is in the project's root directory and your virtual environment is active.
    -   Run the Streamlit application:
        ```bash
        streamlit run app.py
        ```
    -   Your web browser should open with the RegGuard AI interface.
    -   In the sidebar, click the **"Build Vector Store"** button. This will process the files in your `documents` folder. Wait for the success message.

2.  **Start Asking Questions**:
    -   Once the vector store is built, you can use the chat input at the bottom of the page to ask your questions.
    -   The AI will process your query and return an answer along with the sources it used.

The vector store is persistent. You only need to rebuild it if you add, remove, or change the files in the `documents` folder.
