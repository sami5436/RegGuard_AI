from typing import Dict, Any

def retrieve_documents(state: Dict, retriever) -> Dict[str, Any]:
    """Retrieves documents from the vector store."""
    # it uses a retriever object that is initialized with the vector store
    # the retriever will search for relevant documents based on the question
    print("---RETRIEVING DOCUMENTS---")
    question = state['question']
    documents = retriever.invoke(question)
    return {"documents": documents}
