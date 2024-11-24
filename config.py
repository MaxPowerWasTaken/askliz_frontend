import os
import google.generativeai as genai
import streamlit as st
from lancedb.rerankers import CrossEncoderReranker

# CONSTANTS
CLOUD_DB_URI = "db://askliz-doc-db-8qjgop"
N_RESULTS_RETRIEVED = 10  
N_TOP_RERANKED_RESULTS_TO_LLM = 3

# Load secrets
LANCEDB_API_KEY = st.secrets["lancedb"]["api_key"]
GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
COHERE_API_KEY = st.secrets["cohere"]["api_key"]
ANTHROPIC_API_KEY = st.secrets["anthropic"]["api_key"]
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

# Load API Keys into environment (superfluous in prod but helps with local dev)
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
os.environ['COHERE_API_KEY'] = COHERE_API_KEY
os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# see all available with litellm.utils.get_valid_models()
SELECTED_LITELLM_MODEL_OPTIONS = ['gemini/gemini-1.5-pro-latest', 
                                  'gemini/gemini-1.5-flash-latest',
                                  'claude-3-5-sonnet-20241022',
                                  'claude-3-5-sonnet-20240620',
                                  'claude-3-5-haiku-20241022',
                                  'command-r-plus-08-2024',
                                  'command-light',
                                  'gpt-3.5-turbo-16k-0613',
                                  'gpt-4-turbo-2024-04-09',
                                  'chatgpt-4o-latest',
                                  'gpt-4o',
                                  'gpt-4o-mini',
                                  'o1-mini-2024-09-12']

### USER CONFIGURABLE PARAMS ###
DEFAULT_FINAL_LLM_MODEL = "gemini/gemini-1.5-pro-latest"



"""# Load & Configure LLM to craft final response to user ('G' in RAG):
generation_config = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}
genai.configure(api_key=GEMINI_API_KEY)
final_llm = genai.GenerativeModel(FINAL_LLM_MODEL, generation_config=generation_config)
"""
