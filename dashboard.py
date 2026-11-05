"""Fraud reviewer chat and architecture pages.

    .venv\\Scripts\\python.exe -m streamlit run dashboard.py
"""

import streamlit as st

from architecture_ui import architecture_page
from reviewer_ui import apply_theme, chat_page

st.set_page_config(page_title="Fraud reviewer", page_icon="▣", layout="wide", initial_sidebar_state="expanded")
apply_theme()

if "page" not in st.session_state:
    st.session_state.page = "Review"

review, architecture, _rest = st.columns([1.25, 1.6, 4.2])
if review.button("Review", type="primary" if st.session_state.page == "Review" else "tertiary", width="stretch"):
    st.session_state.page = "Review"
    st.rerun()
if architecture.button("Architecture", type="primary" if st.session_state.page == "Architecture" else "tertiary", width="stretch"):
    st.session_state.page = "Architecture"
    st.rerun()

if st.session_state.page == "Architecture":
    architecture_page()
else:
    chat_page()
