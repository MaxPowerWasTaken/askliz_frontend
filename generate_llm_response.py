from typing import List
from google.generativeai import GenerativeModel

def generate_response(
    query: str, 
    context_passages: List[str], 
    model: GenerativeModel,
    debug: bool = False
) -> str:
    """
    Generate a response using the provided LLM based on the query and retrieved context.
    
    Parameters
    ----------
    query : str
        The user's original question
    context_passages : List[Dict[str, Any]]
        Retrieved and reranked passages from the vector database
    model : GenerativeModel
        The Gemini model instance to use for generation
        
    Returns
    -------
    str
        The generated response
    """
    # Prepare the context from retrieved passages
    context = "\n\n".join([f"Context {i+1}:\n{result['text']}" 
                          for i, result in enumerate(context_passages)])
    
    prompt = f"""Based on the following context, please answer the question. 
    Use only the information provided in the context. If you cannot answer the question 
    based on the context alone, please say so.

    Context:
    {context}

    Question: {query}

    Answer:
    """
    if debug:
        print(prompt)

    response = model.generate_content(prompt)
    return response.text