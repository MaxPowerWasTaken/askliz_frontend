import lancedb
import streamlit as st

from lancedb.rerankers import CohereReranker

from config import (LANCEDB_API_KEY,
                    COHERE_API_KEY, 
                    CLOUD_DB_URI, 
                    N_RESULTS_RETRIEVED, 
                    N_TOP_RERANKED_RESULTS_TO_LLM,
                    final_llm)
from generate_llm_response import generate_response

def query_documents(tbl: lancedb.table,
                   query: str, 
                   num_results_retrieved: int = N_RESULTS_RETRIEVED,
                   num_results_to_llm: int = N_TOP_RERANKED_RESULTS_TO_LLM):
    """Query the document database."""
    reranker = CohereReranker(model_name="rerank-english-v2.0", api_key=COHERE_API_KEY)
    results = (tbl.search(query, query_type="hybrid")
               .limit(num_results_retrieved)
               .rerank(reranker=reranker)
               ).to_list()[:num_results_to_llm]

    final_response = generate_response(query, results, model=final_llm, debug=False)

    return final_response

def main():
    st.title("Ask Liz about the January 6 Select Committee's Findings")
    
    # Main query interface
    query = st.text_input("Enter your question or query")
    
    if query:
        tbl = lancedb.connect(CLOUD_DB_URI, api_key=LANCEDB_API_KEY).open_table("document_chunks")        
        response = query_documents(tbl, query)
        st.write(response)

if __name__ == "__main__":
    main()