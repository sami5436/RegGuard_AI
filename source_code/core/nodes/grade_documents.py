from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def grade_documents(state: Dict, llm) -> Dict[str, Any]:
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
    grader_chain = prompt | llm | StrOutputParser()
    
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
