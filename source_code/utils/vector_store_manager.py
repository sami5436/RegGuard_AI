import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorStoreManager:
    """Manages loading documents and creating/loading the FAISS vector store."""
    def __init__(self, docs_path: str, vector_store_path: str, embeddings):
        self.docs_path = docs_path
        self.vector_store_path = vector_store_path
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vector_store = None
        self.retriever = None

    def _is_ready(self) -> bool:
        """Checks if the vector store files exist and are not empty."""
        faiss_path = os.path.join(self.vector_store_path, "index.faiss")
        pkl_path = os.path.join(self.vector_store_path, "index.pkl")
        # Ensure both files exist and the main index file has content.
        return (os.path.exists(faiss_path) and 
                os.path.exists(pkl_path) and 
                os.path.getsize(faiss_path) > 0)

    def _load_documents(self) -> List:
        """Loads PDF and TXT documents from the specified directory."""
        print("Loading PDF documents...")
        pdf_loader = DirectoryLoader(
            self.docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader,
            show_progress=True, use_multithreading=True, silent_errors=True
        )
        pdf_docs = pdf_loader.load()

        print("Loading TXT documents...")
        txt_loader = DirectoryLoader(
            self.docs_path, glob="**/*.txt", loader_cls=TextLoader,
            show_progress=True, silent_errors=True
        )
        txt_docs = txt_loader.load()
        
        return pdf_docs + txt_docs

    def _split_documents(self, documents: List) -> List:
        """Splits documents into chunks."""
        return self.text_splitter.split_documents(documents)

    def build(self):
        """Builds the FAISS vector store from documents."""
        print("Loading documents...")
        documents = self._load_documents()
        if not documents:
            raise ValueError("No documents found. Please add PDF or TXT files to the 'documents' folder.")
            
        print(f"Loaded {len(documents)} documents.")
        
        print("Splitting documents into chunks...")
        chunks = self._split_documents(documents)
        print(f"Created {len(chunks)} document chunks.")
        
        print("Creating FAISS vector store...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.vector_store_path)
        print(f"Vector store saved to {self.vector_store_path}")
        self.retriever = self.vector_store.as_retriever()

    def load(self):
        """Loads an existing FAISS vector store if it's ready."""
        if not self._is_ready():
            raise FileNotFoundError("Vector store is not ready or is corrupted. Please click 'Build Vector Store'.")
        
        print("Loading existing vector store...")
        self.vector_store = FAISS.load_local(
            self.vector_store_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_store.as_retriever()
        print("Successfully loaded vector store.")


    def get_retriever(self):
        """Returns the retriever, loading the vector store if necessary."""
        if self.retriever is None:
            self.load()
        return self.retriever
