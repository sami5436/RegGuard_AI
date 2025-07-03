from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_answer(state: Dict, llm) -> Dict[str, Any]:
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
    rag_chain = prompt | llm | StrOutputParser()
    
    context_str = "\n\n".join([d.page_content for d in documents])
    generation = rag_chain.invoke({"question": question, "context": context_str})
    return {"generation": generation}
