"""
Chat with Uber's 2024 Annual Report using the fine-tuned model or the RAG pipeline.

Needs the adapters from Notebook 02 and the Weaviate index from Notebook 03.
Run from the repo root:  uv run streamlit run app.py
"""
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "notebooks"))

st.set_page_config(page_title="Ledger-Mind Chat")


@st.cache_resource(show_spinner="Loading the model and RAG index (first start takes a minute)...")
def load_systems():
    # Loaded once per server: the base model, adapters and Weaviate are shared by every chat
    import inference_utils
    return {"RAG": inference_utils.query_rag, "Fine-tuned": inference_utils.query_finetuned}


systems = load_systems()

with st.sidebar:
    mode = st.selectbox("Answer with", list(systems))
    st.caption(
        "**RAG** retrieves passages from the report and answers from them.\n\n"
        "**Fine-tuned** answers from what the model learned in training, with no retrieval.\n\n"
        "Both use the same Llama-3.2-3B base model. Each question is answered on its own, "
        "without the earlier chat as context."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []

st.title("Uber 2024 Annual Report Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_answer(message):
    st.markdown(message["content"])
    st.caption(f"{message['mode']} · {message['latency']:.1f}s")
    if message["sources"]:
        with st.expander(f"Sources ({len(message['sources'])})"):
            for source in message["sources"]:
                st.markdown(f"**Page {source['page']}** · chunk {source['chunk_id']}")
                st.text(source["text"][:600])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            show_answer(message)

if question := st.chat_input("Ask about Uber's 2024 Annual Report"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(f"{mode} is answering..."):
            start = time.time()
            if mode == "RAG":
                result = systems["RAG"](question, return_sources=True)
                answer, sources = result["answer"], result["sources"]
            else:
                answer, sources = systems["Fine-tuned"](question), []
            latency = time.time() - start

        message = {"role": "assistant", "content": answer, "mode": mode, "latency": latency, "sources": sources}
        show_answer(message)

    st.session_state.messages.append(message)
