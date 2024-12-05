import lancedb
import litellm
import pandas as pd
import streamlit as st

from lancedb.rerankers import RRFReranker, CohereReranker, LinearCombinationReranker, CrossEncoderReranker, ColbertReranker

from config import (LANCEDB_API_KEY,
                    COHERE_API_KEY, 
                    CLOUD_DB_URI, 
                    N_RESULTS_RETRIEVED, 
                    N_TOP_RERANKED_RESULTS_TO_LLM,
                    DEFAULT_FINAL_LLM_MODEL,
                    DEFAULT_TEMPERATURE,
                    MODEL_OPTIONS)

from generate_llm_response import generate_response
from schemas import RetrievedDocumentChunk

def get_most_relevant_chunks(tbl: lancedb.table,
                   query: str, 
                   num_results_retrieved: int = N_RESULTS_RETRIEVED,
                   num_results_to_llm: int = N_TOP_RERANKED_RESULTS_TO_LLM,
                   final_rr = CohereReranker(model_name="rerank-english-v3.0", api_key=COHERE_API_KEY),
                   )->list[RetrievedDocumentChunk]:
    """Query the document database."""

    # build hybrid search plan (lazy eval; nothing here til we call .to_list(), etc)
    search_obj = (tbl.search(query, query_type="hybrid")
            .limit(num_results_retrieved)
            .rerank(reranker=final_rr))

    # execute the search plan, get results as list of RetrievedDocumentChunk obj fr pydantic schema
    res_list = search_obj.to_list()[:num_results_to_llm]
    results = [RetrievedDocumentChunk(**el, 
                                      relevance_score=el['_relevance_score']) for el in res_list]

    # ensure we always return results w/ top relevance scores first
    results.sort(key=lambda x: x.relevance_score, reverse=True)

    return results


def get_selected_rr(selected_reranker):
    match selected_reranker:
        case "CohereReranker":
            return CohereReranker(model_name="rerank-english-v3.0", api_key=COHERE_API_KEY)
        case "LinearCombinationReranker":
            return LinearCombinationReranker()  # default (vector-score) weight=0.7
        case "CrossEncoderReranker":
            return CrossEncoderReranker()
        case "ColbertReranker":
            return ColbertReranker()
        case _:
            raise ValueError(f"Unknown reranker type: {selected_reranker}")
            
def main():
    st.title("Ask Liz about the January 6 Select Committee's Findings")

    with st.sidebar:
        # Input Number of Chunks Retrieved by 1st Stage; range: 5-15
        num_results_retrieved = st.number_input(label="Number of Chunks Retrieved (1st Stage)",
                                                min_value=5, max_value=15, value=N_RESULTS_RETRIEVED,
                                                help="Number of document chunks retrieved from the document database (1st stage; before re-ranking)")

        # Input Type of Reranker
        selected_reranker = st.selectbox(label="Type of Reranker",
                                         options=["CohereReranker", "LinearCombinationReranker", "CrossEncoderReranker", "ColbertReranker"],
                                         index=0,
                                         help="Type of reranker to use for re-ranking the retrieved document chunks")
        final_rr = get_selected_rr(selected_reranker)

        # Input Number of Top-Reranked Chunks to LLM; range: 1-5
        num_results_to_llm = st.number_input(label="Number of Top-Reranked Chunks to LLM",
                                             min_value=1, max_value=10, value=N_TOP_RERANKED_RESULTS_TO_LLM,
                                             help="Number of top-ranked document chunks to send to LLM for final answer formulation")

        selected_llm_model = st.selectbox(label="Final LLM Model",
                                          options=MODEL_OPTIONS, 
                                          index=MODEL_OPTIONS.index(DEFAULT_FINAL_LLM_MODEL),
                                          help="LLM Model wihch takes the most-relevant retrieved document chunks, and formulates a final answer",
    )
        temperature = st.number_input(label="Temperature",
                                      min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE,
                                      help="Temperature parameter for LLM model (lower value for more deterministic)")

        mode_help_txt = """Show intermediate results at each step of RAG frontend pipeline 
        (downstream of document ingestion/parsing/chunking/indexing into document-db)"""
        show_steps = st.toggle("Introspection Mode", value=False, help=mode_help_txt)


    # Main query interface
    query = st.text_input("Enter your question or query")
    
    if query:
        tbl = lancedb.connect(CLOUD_DB_URI, api_key=LANCEDB_API_KEY).open_table("document_chunks")
        retrieved_chunks = get_most_relevant_chunks(tbl, 
                                                    query, 
                                                    num_results_retrieved=num_results_retrieved,
                                                    num_results_to_llm=num_results_to_llm,
                                                    final_rr=final_rr,
                                                    )
        final_response, final_prompt = generate_response(query, 
                                                         retrieved_chunks, 
                                                         llm_name=selected_llm_model,
                                                         temperature=temperature,
                                                         )
        
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