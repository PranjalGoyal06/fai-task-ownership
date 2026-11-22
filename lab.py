"""Unified entry point for the historical experiment and synthetic simulator."""
import streamlit as st

st.set_page_config(page_title="Task ownership lab", layout="wide")
page = st.navigation([
    st.Page("app/lab/historical.py", title="Historical replay", default=True),
    st.Page("app/lab/setup.py", title="New repository experiment", url_path="setup"),
    st.Page("app.py", title="Synthetic simulation", url_path="simulation"),
])
page.run()
