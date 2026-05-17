import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import logging
logging.set_verbosity_error()

import streamlit as st
from backend.generation.generate import rewrite_before_gen
from backend.retrieval.retrieve import get_all_titles
st.set_page_config(
    page_title="PaperPro",
    layout="wide"
)

@st.cache_data
def fetch_papers():
    return get_all_titles()

with st.sidebar:
    st.header("Available Papers")
    paper_titles=fetch_papers()
    for p in paper_titles:
        st.write(p)

st.title("PaperPro - Your Personal Research Assistant")

query = st.text_input(
    "Ask a research question"
)

if st.button("Generate Answer"):

    with st.spinner("Thinking..."):

        result = rewrite_before_gen(query)

        if result is None:
            st.error("Generation Failed")
        else:
            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources")

            final_sources=set(result["sources"])

            for i, source in enumerate(final_sources):
                st.markdown(f"**{i+1}. {source}**")
                

        