#!/usr/bin/env bash
set -e
cd '/home/jimmy_linux/anaconda_projects/nlp/Social media emotion analyzer project'
exec /home/jimmy_linux/venvs/ml_env/bin/python -m streamlit run app.py --server.port 8501 --server.headless true > streamlit_full.out.log 2> streamlit_full.err.log
