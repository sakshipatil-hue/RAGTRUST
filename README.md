# RAG Reliability Engineer — Advanced Dashboard

A dark, technical Streamlit observability dashboard for the pilot scorecard.

## Run locally (Windows)
1. Extract the ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal in that folder.
4. Run:

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If `py` is unavailable, use `python` instead.

## Data
The included `rag_reliability_final_scorecard.csv` is the pilot scorecard.
You can also upload a compatible CSV from the sidebar.

## Important interpretation
- Retrieval Hit@3 and MRR are baseline retrieval metrics.
- Semantic similarity and whole-answer NLI are baseline diagnostics.
- Agent Action and claim counts are from the final v3 agent run.
- The dashboard deliberately labels these stages separately.
- The pilot has only four questions; do not present the metrics as production-level evidence.
