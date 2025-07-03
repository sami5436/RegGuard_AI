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
    # prompttemplate helps us create a prompt that the LLM will use to generate an answer
    prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])
    # rag_chain is an object that combines the prompt with the LLM and an output parser
    rag_chain = prompt | llm | StrOutputParser()
    # We join the page contents of the documents to create a context string
    context_str = "\n\n".join([d.page_content for d in documents])
    # we then provide the question and context to the rag_chain to get the answer
    generation = rag_chain.invoke({"question": question, "context": context_str})
    return {"generation": generation}
