# Askliz (frontend)
This is the repo for the frontend code for the Askliz project, a RAG chatbot for questions about the findings of the House Select Committee on the January 6th Capitol Insurrection. 

The frontend uses streamlit and is available online at askliz.streamlit.app

# Local Development
In a virtual environment, run the following commands:
```bash
pip install --upgrade pip pip-tools wheel
pip install -r requirements.txt
```
...then to view the effects of any local development changes, launch a local version of the streamlit app with:

```bash
PYTHONPATH=. streamlit run app.py
```