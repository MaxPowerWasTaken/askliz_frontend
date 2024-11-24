import litellm
import pandas as pd

from typing import Tuple
from google.generativeai import GenerativeModel
def generate_response(
    query: str, 
    context_passages: pd.DataFrame, 
    llm_name: str,
) -> Tuple[str, str]:
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
    context_passages_list = context_passages['text'].tolist()
    context = "\n\n".join([f"Context {i+1}:\n{text_chunk}" 
                          for i, text_chunk in enumerate(context_passages_list)])
    
    prompt = f"""Based on the following context, please answer the question. 
    Use only the information provided in the context. If you cannot answer the question 
    based on the context alone, please say so.

    Context:
    {context}

    Question: {query}

    Answer:
    """
    messages = [{"content": prompt, "role": "user"}]
    response = litellm.completion(model=llm_name, messages=messages)#, api_key=GEMINI_API_KEY)
    response_text = response['choices'][0]['message']['content']

    return (response_text, prompt)