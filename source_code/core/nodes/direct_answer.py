from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_direct_answer(state: Dict, llm) -> Dict[str, Any]:
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
    direct_chain = prompt | llm | StrOutputParser()
    generation = direct_chain.invoke({"question": question})
    return {"generation": generation, "documents": []}
