from typing import Dict, Any

def retrieve_documents(state: Dict, retriever) -> Dict[str, Any]:
    """Retrieves documents from the vector store."""
    print("---RETRIEVING DOCUMENTS---")
    question = state['question']
    documents = retriever.invoke(question)
    return {"documents": documents}
