from typing import List, Dict, Any, TypedDict
from functools import partial

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langgraph.graph import END, StateGraph

from source_code.utils.vector_store_manager import VectorStoreManager
from source_code.core.nodes.check_relevance import check_relevance
from source_code.core.nodes.retrieve import retrieve_documents
from source_code.core.nodes.grade_documents import grade_documents
from source_code.core.nodes.generate import generate_answer
from source_code.core.nodes.direct_answer import generate_direct_answer

# --- State Definition for LangGraph ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Any]
    is_relevant: bool

# --- Main RAG System Class ---
class RAGSystem:
    def __init__(self, docs_path: str, vector_store_path: str, model_name: str = "llama3"):
        # Initialize LLM and Embeddings
        self.llm = Ollama(model=model_name, temperature=0)
        self.embeddings = OllamaEmbeddings(model=model_name)
        
        # Initialize Vector Store Manager
        self.vector_store_manager = VectorStoreManager(
            docs_path=docs_path,
            vector_store_path=vector_store_path,
            embeddings=self.embeddings
        )
        
        # **FIX**: Workflow is not built at initialization anymore.
        # It will be built lazily when first needed.
        self.workflow = None

    def _initialize_workflow(self):
        """
        Initializes the LangGraph workflow on-demand.
        This prevents the app from crashing on startup by not trying to load a
        vector store that doesn't exist yet.
        """
        if self.workflow is None:
            print("--- Initializing workflow for the first time ---")
            self.workflow = self._build_graph()

    def _build_graph(self) -> Any:
        """Builds and compiles the LangGraph workflow."""
        workflow = StateGraph(GraphState)

        # Bind dependencies to node functions
        # the partial function allows us to bind the llm 
        # to the functions so that they can be used without passing them explicitly
        bound_check_relevance = partial(check_relevance, llm=self.llm)
        bound_retrieve_documents = partial(retrieve_documents, retriever=self.vector_store_manager.get_retriever())
        bound_grade_documents = partial(grade_documents, llm=self.llm)
        bound_generate_answer = partial(generate_answer, llm=self.llm)
        bound_direct_answer = partial(generate_direct_answer, llm=self.llm)

        # Define nodes
        workflow.add_node("check_relevance", bound_check_relevance)
        workflow.add_node("retrieve", bound_retrieve_documents)
        workflow.add_node("grade_documents", bound_grade_documents)
        workflow.add_node("generate", bound_generate_answer)
        workflow.add_node("direct_answer", bound_direct_answer)
        workflow.add_node("handle_no_generation", self.handle_no_generation)

        # we're saying start at the relevance
        workflow.set_entry_point("check_relevance")
        
        
        # this coniditional edge will check if the question is relevant
        # if it is, it will go to the retrieve node
        # if it is not, it will go to the direct_answer node
        workflow.add_conditional_edges("check_relevance", self.decide_to_retrieve)
        
        # if the question is relevant, we retrieve documents
        workflow.add_edge("retrieve", "grade_documents")
        
        # if the documents are found, we grade them
        workflow.add_conditional_edges("grade_documents", self.decide_to_generate)
        
        # if the documents are relevant, we generate an answer
        workflow.add_edge("generate", END)
        
        # if the original question is not relevant, we generate a direct answer
        workflow.add_edge("direct_answer", END)
        
        # if no documents are found, we handle the case where generation is not possible
        workflow.add_edge("handle_no_generation", END)

        return workflow.compile()

    # --- Conditional Edge Logic ---
    def decide_to_retrieve(self, state: Dict) -> str:
        return "retrieve" if state.get('is_relevant') else "direct_answer"

    def decide_to_generate(self, state: Dict) -> str:
        return "generate" if state.get('documents') else "end_without_generation"
    
    def handle_no_generation(self, state: Dict) -> Dict[str, Any]:
        return {"generation": "I could not find any relevant documents to answer your question."}

    def query(self, question: str) -> Dict[str, Any]:
        """Runs a query through the RAG workflow, initializing it if necessary."""
        # **FIX**: Ensure the workflow is initialized before running a query.
        self._initialize_workflow()
        inputs = {"question": question, "documents": [], "generation": "", "is_relevant": False}
        return self.workflow.invoke(inputs)

    def build_vector_store(self):
        """Builds the vector store and resets the workflow to be rebuilt with new data."""
        self.vector_store_manager.build()
        # **FIX**: Reset workflow. It will be re-initialized on the next query,
        # ensuring it uses the newly built vector store.
        self.workflow = None
