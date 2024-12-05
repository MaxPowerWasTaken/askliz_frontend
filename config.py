import os
import google.generativeai as genai
import streamlit as st
from lancedb.embeddings import get_registry
from lancedb.rerankers import CrossEncoderReranker

# CONSTANTS
CLOUD_DB_URI = "db://askliz-doc-db-8qjgop"
EMBEDDING_MODEL = ("sentence-transformers", "BAAI/bge-small-en-v1.5")  # used for vector field in CLOUD_DB_URI 
embedding_model = get_registry().get(EMBEDDING_MODEL[0]).create(name=EMBEDDING_MODEL[1])
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

# Restricting model options now that we care more about structured output,
# which OpenAI supports in a particularly clean and reliable way.
# (pricing info below from https://openai.com/api/pricing/ as of 11/26/2024)
MODEL_OPTIONS = ['gpt-4o',                  # $2.50 / $10.00 price per M input/output tokens
                 'gpt-4o-2024-11-20',       # $2.50 / $10.00 
                 "gpt-4o-mini",             # $0.150 / $0.600
                 "gpt-4o-mini-2024-07-18"   # $0.150 / $0.600
                 "o1-mini",                 # $3.00 / $12.00 
                 ]


### USER CONFIGURABLE PARAMS ###
DEFAULT_FINAL_LLM_MODEL = "gpt-4o-mini"  # we should be able to get by w/ smallest model avaialable for this step
DEFAULT_TEMPERATURE = 0.3