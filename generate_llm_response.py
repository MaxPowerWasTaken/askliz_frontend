from typing import List, Tuple
from pydantic import BaseModel, Field, validator
import pandas as pd
from openai import OpenAI

from schemas import RetrievedDocumentChunk
# TODO: improve with claude answer: https://claude.ai/chat/14f056ef-b7b4-4289-a4e1-cd2a62db5b46

# https://python.useinstructor.com/blog/2023/11/18/validate-citations/


# Setting up some Pydantic Models for our LLM Structured Output:
# A response should contain 1-3 points, and each point should cite to a quote/subsection

class QuotedSection(BaseModel):
    quote: str = Field(..., description="A relevant quote from the source text, potentially using ... for omissions")
    section_title: str = Field(..., description="The section title where this quote was found")

class NarrativeResponse(BaseModel):
    content: str = Field(
        ..., 
        description="A cohesive narrative response that naturally incorporates quoted material using "
                   "phrases like 'According to...', 'As stated in...', etc. Quotes should be wrapped "
                   "in quotation marks and followed by section references in parentheses."
    )
    quotes: List[QuotedSection] = Field(
        ..., 
        description="List of all quotes used in the narrative, in order of appearance"
    )


def format_structured_response(response):
    """
    Parses the StructuredResponse object and formats it into a readable string.
    
    Args:
        response (object): The response object from the chat completion.

    Returns:
        str: A nicely formatted string representation of the response.
    """
    try:
        points = response.choices[0].message.parsed.points  # Extract the points from the response
        formatted_output = "Key Points from the Response:\n\n"
        
        for i, point in enumerate(points, 1):
            formatted_output += f"{i}. {point.title}\n"
            formatted_output += f"   - Content: {point.content}\n"
            formatted_output += f"   - Citation:\n"
            formatted_output += f"     * Quote: \"{point.citation.quote}\"\n"
            formatted_output += f"     * Section Title: {point.citation.section_title}\n\n"
        
        return formatted_output.strip()
    except AttributeError as e:
        return f"Error parsing the response: {e}"

def generate_response(
    query: str, 
    context_passages: list[RetrievedDocumentChunk],
    llm_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
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
    print(context_passages)
    context = "\n\n".join([f"""chunk_idx: {c.chunk_idx}
                            relevance_score: {c.relevance_score}
                            section_title: {c.section_title}
                            text: {c.text}
                            """
                          for c in context_passages])
    
    prompt = f"""You are a thoughtful research assistant, providing answers to questions based on 
    insightful reviews of source material, and you cite specific passages that back up your claims.

    Below, I will present you with a question, followed by a list of source materials as
    context. The code/schema that defines the source materials is as follows:

    context =   chunk_idx: c.chunk_idx
                relevance_score: c.relevance_score
                section_title: c.section_title
                text: c.text  
        ... for c in most_relevant_document_chunks

    The 'relevance_score' in particular is very reliable (higher scores are more relevant); 
    if a particular source material passage has a significantly higher/lower relevance score 
    than others, please increase/decrease your focus on it in your answer accordingly

    It is very important to formulate your answer to utilize direct quote(s) from one or (ideally) more 
    of the source materials. After a direct quote, include a citation in [] brackets with the 
    `section_title` of the source material. For brevity, direct quotes can
    omit unnecessary words with ellipsis ('...'), between two (or more) key substrings
    from a source material passage.

    Below is the QUESTION and SOURCE MATERIALS, as described above:
    ---------------------------------------------
    QUESTION: {query}

    SOURCE MATERIALS:
    {context}
    """
    messages = [{"content": prompt, "role": "user"}]
    client = OpenAI()
    response = client.beta.chat.completions.parse(
        model=llm_name,
        messages=messages,
        temperature=temperature,
        response_format=NarrativeResponse
    )

    final_user_response = response.choices[0].message.parsed.content #format_structured_response(response)

    return (final_user_response, prompt)