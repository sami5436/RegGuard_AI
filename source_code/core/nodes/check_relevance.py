from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def check_relevance(state: Dict, llm) -> Dict[str, Any]:
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
    relevance_checker = prompt | llm | StrOutputParser()
    
    relevance_response = relevance_checker.invoke({"question": question})
    if "yes" in relevance_response.lower():
        print("---QUESTION IS RELEVANT---")
        return {"is_relevant": True}
    else:
        print("---QUESTION IS NOT RELEVANT---")
        return {"is_relevant": False}
