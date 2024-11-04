import google.generativeai as genai
import streamlit as st
from lancedb.rerankers import CrossEncoderReranker

# CONSTANTS
CLOUD_DB_URI = "db://askliz-doc-db-8qjgop"
N_RESULTS_RETRIEVED = 10  
N_TOP_RERANKED_RESULTS_TO_LLM = 3

# Load secrets
GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
LANCEDB_API_KEY = st.secrets["lancedb"]["api_key"]
COHERE_API_KEY = st.secrets["cohere"]["api_key"]

# Load & Configure LLM to craft final response to user ('G' in RAG):
FINAL_LLM_MODEL = "gemini-1.5-pro"
generation_config = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}

genai.configure(api_key=GEMINI_API_KEY)
final_llm = genai.GenerativeModel(FINAL_LLM_MODEL, generation_config=generation_config)
