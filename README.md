# RegGuard AI: RAG for Compliance Documents

RegGuard AI is an interactive, self-hosted platform designed to help you navigate complex regulatory documents, specifically tailored for oil & gas emissions compliance. It uses a powerful Retrieval-Augmented Generation (RAG) pipeline to provide answers to your questions, citing the specific sources from your documents.

This version of the project has been refactored into a fully modular structure to improve maintainability and scalability.


## Setup and Installation

Follow these steps to get RegGuard AI running on your local machine.

### Step 1: Install Ollama

You must have Ollama installed and running.

1.  Go to [https://ollama.com/](https://ollama.com/) and download the application for your operating system.
2.  After installation, open your terminal and pull a model. We recommend `llama3`.
    ```bash
    ollama run llama3
    ```
3.  Leave Ollama running in the background.

### Step 2: Project Setup

1.  **Create the directory structure** as shown above. This is crucial for the modular imports to work correctly.
2.  Place all the provided `.py` files into their corresponding directories.
3.  Create empty `__init__.py` files in the `source_code`, `source_code/app`, `source_code/core`, `source_code/core/nodes`, and `source_code/utils` directories. This tells Python to treat them as packages.
4.  **Create a Python virtual environment**.
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```
5.  **Install the required Python packages**:
    ```bash
    pip install -r requirements.txt
    ```

### Step 3: Add Documents

1.  Place your regulatory documents (`.pdf` or `.txt`) inside the `documents` folder.

## How to Run the Application

1.  **Navigate to the project's root directory** in your terminal.
2.  **Run the Streamlit application using the `main.py` file**:
    ```bash
    streamlit run main.py
    ```
3.  Your web browser will open with the RegGuard AI interface.
4.  In the sidebar, click **"Build Vector Store"** to process your documents.
5.  Once built, start asking questions in the chat interface.
