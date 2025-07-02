import os
from typing import List, Dict, Any, TypedDict

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

# --- State Definition for LangGraph ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        generation: The LLM's generated answer.
        documents: A list of retrieved documents.
        is_relevant: A boolean indicating if the question is relevant to the document context.
    """
    question: str
    generation: str
    documents: List[Any]
    is_relevant: bool


# --- Main RAG System Class ---
class RAGSystem:
    def __init__(self, docs_path: str, vector_store_path: str, model_name: str = "llama3"):
        self.docs_path = docs_path
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        
        # Initialize components
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = OllamaEmbeddings(model=self.model_name)
        self.llm = Ollama(model=self.model_name, temperature=0)
        self.vector_store = None
        self.retriever = None
        
        # Compile the LangGraph workflow
        self.workflow = self._build_graph()

    def _load_documents(self):
        """Loads PDF and TXT documents from the specified directory."""
        print("Loading PDF documents...")
        pdf_loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True
        )
        pdf_docs = pdf_loader.load()

        print("Loading TXT documents...")
        txt_loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
            silent_errors=True
        )
        txt_docs = txt_loader.load()
        
        all_docs = pdf_docs + txt_docs
        return all_docs

    def _split_documents(self, documents: List):
        """Splits documents and adds metadata."""
        chunks = self.text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
        return chunks

    def build_vector_store(self):
        """Builds the FAISS vector store from documents."""
        print("Loading documents...")
        documents = self._load_documents()
        if not documents:
            print("No documents found. Please add PDF or TXT files to the 'documents' folder.")
            return
            
        print(f"Loaded {len(documents)} documents.")
        
        print("Splitting documents into chunks...")
        chunks = self._split_documents(documents)
        print(f"Created {len(chunks)} document chunks.")
        
        print("Creating FAISS vector store...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.vector_store_path)
        print(f"Vector store saved to {self.vector_store_path}")
        self.retriever = self.vector_store.as_retriever()

    def _load_vector_store(self):
        """Loads an existing FAISS vector store."""
        if os.path.exists(self.vector_store_path):
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vector_store.as_retriever()
            print("Successfully loaded vector store.")
        else:
            raise FileNotFoundError("Vector store not found. Please build it first.")

    # --- LangGraph Nodes ---

    def check_relevance(self, state: GraphState) -> Dict[str, Any]:
        """Checks if the question is relevant to the document context."""
        print("---CHECKING QUESTION RELEVANCE---")
        question = state['question']
        prompt_template = """
        You are an expert in oil and gas regulations. Your task is to determine if a user's question is related to 'oil and gas emissions' or 'emissions' in general.
        Answer with a simple 'yes' or 'no'.

        User question: "{question}"

        Is this question related to emissions? (yes/no):
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
        relevance_checker = prompt | self.llm | StrOutputParser()
        
        relevance_response = relevance_checker.invoke({"question": question})
        if "yes" in relevance_response.lower():
            print("---QUESTION IS RELEVANT---")
            return {"is_relevant": True}
        else:
            print("---QUESTION IS NOT RELEVANT---")
            return {"is_relevant": False}

    def retrieve_documents(self, state: GraphState) -> Dict[str, Any]:
        """Retrieves documents from the vector store."""
        print("---RETRIEVING DOCUMENTS---")
        question = state['question']
        documents = self.retriever.invoke(question)
        return {"documents": documents}

    def generate_answer(self, state: GraphState) -> Dict[str, Any]:
        """Generates an answer using the LLM with context."""
        print("---GENERATING ANSWER WITH CONTEXT---")
        question = state['question']
        documents = state['documents']
        
        prompt_template = """
        You are an assistant for question-answering tasks for oil and gas emissions compliance.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Be concise and provide the answer based only on the provided context.

        Question: {question}
        Context: {context}

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])
        
        rag_chain = prompt | self.llm | StrOutputParser()
        
        context_str = "\n\n".join([d.page_content for d in documents])
        generation = rag_chain.invoke({"question": question, "context": context_str})
        return {"generation": generation}

    def generate_direct_answer(self, state: GraphState) -> Dict[str, Any]:
        """Generates a direct answer when the question is not relevant to the documents."""
        print("---GENERATING DIRECT ANSWER---")
        question = state['question']
        
        prompt_template = """
        You are a helpful assistant. A user has asked a question that is not related to the provided regulatory documents.
        Provide a direct answer to their question, prefaced with the following disclaimer:
        "This question does not seem to be about emissions compliance, but here is an answer:"

        User Question: {question}

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
        direct_chain = prompt | self.llm | StrOutputParser()
        generation = direct_chain.invoke({"question": question})
        return {"generation": generation, "documents": []} # Ensure documents are empty

    def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """Grades the relevance of retrieved documents."""
        print("---CHECKING DOCUMENT RELEVANCE---")
        question = state['question']
        documents = state['documents']
        
        prompt_template = """
        You are a grader assessing the relevance of a retrieved document to a user question about emissions.
        Give a binary score 'yes' or 'no'. 'yes' means the document is relevant, 'no' means it's not.

        Retrieved document:
        {document_content}

        User question: {question}

        Grade (yes/no):
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["question", "document_content"])
        grader_chain = prompt | self.llm | StrOutputParser()
        
        filtered_docs = []
        for d in documents:
            score = grader_chain.invoke({"question": question, "document_content": d.page_content})
            grade = score.strip().lower()
            if "yes" in grade:
                print(f"Grade is 'yes' for document: {d.metadata['source']}")
                filtered_docs.append(d)
            else:
                print(f"Grade is 'no' for document: {d.metadata['source']}")
        
        return {"documents": filtered_docs}

    # --- LangGraph Conditional Edges ---

    def decide_to_retrieve(self, state: GraphState) -> str:
        """Decides whether to retrieve documents or generate a direct answer."""
        print("---DECIDING TO RETRIEVE---")
        if state['is_relevant']:
            return "retrieve"
        else:
            return "direct_answer"

    def decide_to_generate(self, state: GraphState) -> str:
        """Decides whether to generate an answer or end the process."""
        print("---ASSESSING RELEVANCE & DECIDING TO GENERATE---")
        if not state['documents']:
            return "end_without_generation"
        else:
            return "generate"

    def handle_no_generation(self, state: GraphState) -> Dict[str, Any]:
        """Handles the case where no relevant documents are found."""
        print("---NO RELEVANT DOCUMENTS---")
        return {"generation": "I could not find any relevant documents to answer your question."}

    # --- Build and Compile Graph ---

    def _build_graph(self) -> Any:
        """Builds and compiles the LangGraph workflow."""
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("check_relevance", self.check_relevance)
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("direct_answer", self.generate_direct_answer)
        workflow.add_node("handle_no_generation", self.handle_no_generation)

        # Build graph
        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            self.decide_to_retrieve,
            {
                "retrieve": "retrieve",
                "direct_answer": "direct_answer",
            },
        )
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "generate": "generate",
                "end_without_generation": "handle_no_generation",
            },
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("direct_answer", END)
        workflow.add_edge("handle_no_generation", END)

        # Compile the graph
        return workflow.compile()

    def query(self, question: str) -> Dict[str, Any]:
        """Runs a query through the RAG workflow."""
        if self.retriever is None:
            self._load_vector_store()
            
        inputs = {"question": question, "documents": [], "generation": "", "is_relevant": False}
        result = self.workflow.invoke(inputs)
        return result
