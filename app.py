import lancedb
import pandas as pd
import streamlit as st

from lancedb.rerankers import CohereReranker, RRFReranker, LinearCombinationReranker

from config import (LANCEDB_API_KEY,
                    COHERE_API_KEY, 
                    CLOUD_DB_URI, 
                    N_RESULTS_RETRIEVED, 
                    N_TOP_RERANKED_RESULTS_TO_LLM,
                    final_llm)
from generate_llm_response import generate_response

def get_most_relevant_chunks(tbl: lancedb.table,
                   query: str, 
                   num_results_retrieved: int = N_RESULTS_RETRIEVED,
                   num_results_to_llm: int = N_TOP_RERANKED_RESULTS_TO_LLM,
                   )->pd.DataFrame:
    """Query the document database."""

    # since CohereReranker doesn't support return_score="all", 
    # but also no way to get non-rr scores (e.g. bm25) from hybrid
    # search without a reranker, I use a preliminary reranker for scores introspection
    prelim_rr = RRFReranker(return_score="all")
    prelim_results = (tbl.search(query, query_type="hybrid")
            .limit(num_results_retrieved)
            .rerank(reranker=prelim_rr)
            )
    df0 = prelim_results.to_pandas()

    # final reranked results (in _relevance_score order as determined by reranker)
    final_rr = CohereReranker(model_name="rerank-english-v3.0", api_key=COHERE_API_KEY)
    final_results = prelim_results.rerank(reranker=final_rr).limit(num_results_to_llm)
    
    # joining final reranked results with intermediate distance/bm25 scores
    score_cols = ['_score', '_distance']
    join_cols = ['chunk_idx']
    results_df = final_results.to_pandas().merge(df0[score_cols+join_cols], on=join_cols, how="left")
    assert results_df['_relevance_score'].is_monotonic_decreasing

    return results_df

def main():
    st.title("Ask Liz about the January 6 Select Committee's Findings")

    with st.sidebar:
        mode_help_txt = """Show intermediate results at each step of RAG frontend pipeline 
        (downstream of document ingestion/parsing/chunking/indexing into document-db)"""
        show_steps = st.toggle("Introspection Mode", value=False, help=mode_help_txt)


    # Main query interface
    query = st.text_input("Enter your question or query")
    
    if query:
        tbl = lancedb.connect(CLOUD_DB_URI, api_key=LANCEDB_API_KEY).open_table("document_chunks")        
        retrieved_chunks_df = get_most_relevant_chunks(tbl, query)
        final_response, final_prompt = generate_response(query, retrieved_chunks_df, model=final_llm)
        
        if not show_steps:
            st.write(final_response)

        else:
            tab1, tab2, tab3 = st.tabs(["Final Response", 
                                        "Final Prompt Sent to LLM", 
                                        "Retrieved Doc Chunks w/ Scores"])

            with tab1:
                st.write(final_response)
            with tab2:
                st.write(final_prompt)
            with tab3:
                st.dataframe(retrieved_chunks_df)
if __name__ == "__main__":
    main()