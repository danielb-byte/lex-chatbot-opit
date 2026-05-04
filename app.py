import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

"""Lex Fridman Podcast Chatbot — Streamlit interface.

Author: Daniel Birsan, May 2026.
"""
import os
import subprocess
from pathlib import Path

import requests
import streamlit as st

from langchain_chroma import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ----- config ---------------------------------------------------------------
CSV_NAME    = "podcastdata_dataset.csv"
MIRROR      = "https://github.com/Hritik003/RAG-HYDE-Algorithm-based-Q-A-System.git"
PERSIST_DIR = "./chroma_db"
COLLECTION  = "lex_fridman_podcast"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LIMIT       = 30                       # episodes; set to None for the full 319
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
HF_MODEL    = "microsoft/Phi-3-mini-4k-instruct"

PROMPT = PromptTemplate.from_template(
    """You are answering questions using ONLY the excerpts from the
Lex Fridman Podcast transcripts below. If the answer is not in the excerpts,
say you do not know rather than making something up.

Context:
{context}

Question: {question}
Answer:"""
)

# ----- helpers --------------------------------------------------------------
def ollama_up() -> bool:
    try:
        return requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1).status_code == 200
    except Exception:
        return False


@st.cache_resource(show_spinner="Fetching the dataset (first launch only) …")
def fetch_dataset():
    if not Path(CSV_NAME).exists():
        subprocess.check_call(["git", "clone", "--depth", "1", MIRROR, "./_lexpod"])
        Path("./_lexpod/podcastdata_dataset.csv").rename(CSV_NAME)
    import pandas as pd
    return pd.read_csv(CSV_NAME)


@st.cache_resource(show_spinner="Building / loading the Chroma index …")
def get_vectordb():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True}
    )
    db = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    if db._collection.count() == 0:
        df = fetch_dataset()
        sub = df.copy() if LIMIT is None else df.head(LIMIT).copy()
        sub["source"] = sub.apply(
            lambda r: f"#{int(r['id']):03d} – {r['guest']} – {r['title']}", axis=1
        )
        raw = DataFrameLoader(
            sub[["text", "id", "guest", "title", "source"]],
            page_content_column="text",
        ).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        ).split_documents(raw)
        db.add_documents(chunks)
    return db


def get_llm():
    if ollama_up():
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            base_url=OLLAMA_HOST,
            temperature=0.2,
            num_predict=512,
        ), "Ollama (local llama3.2)"

    token = (
        os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
    )
    if not token:
        st.error(
            "No LLM available. Either start Ollama on this machine, or set "
            "`HUGGINGFACEHUB_API_TOKEN` in Streamlit secrets."
        )
        st.stop()
    from langchain_community.llms import HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(
        repo_id=HF_MODEL,
        huggingfacehub_api_token=token,
        max_new_tokens=512,
        temperature=0.2,
    )
    return llm, f"Hugging Face Inference API ({HF_MODEL})"


# ----- UI -------------------------------------------------------------------
st.set_page_config(
    page_title="Lex Fridman Podcast Chatbot",
    page_icon="🎙️",
    layout="wide",
)
st.title("🎙️  Lex Fridman Podcast Chatbot")
st.caption(
    "Ask anything about the conversations on the Lex Fridman Podcast. "
    "Answers are grounded in episode transcripts via a local Chroma vector "
    "database. Built for the OPIT *Applications in Data Science and AI – Part 2* "
    "reassessment by Daniel Birsan."
)

vectordb = get_vectordb()
llm, backend = get_llm()

with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**Backend:** {backend}")
    st.write(f"**Indexed chunks:** {vectordb._collection.count()}")
    top_k = st.slider("Top-k chunks", 2, 8, 4, 1)
    if st.button("🔄 Reset conversation"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown(
        "**Pipeline:** MiniLM-L6-v2 embeddings → Chroma → "
        "LangChain `RetrievalQA` → LLM."
    )

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": top_k}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

if question := st.chat_input("e.g. What does Elon Musk say about Mars?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching transcripts and thinking …"):
            try:
                result = qa.invoke({"query": question})
            except Exception as e:
                st.error(f"LLM call failed: {e}")
                st.stop()
        answer = result["result"].strip()
        st.markdown(answer)

        seen, sources = set(), []
        for d in result["source_documents"]:
            tag = d.metadata.get("source", "")
            if tag and tag not in seen:
                seen.add(tag); sources.append(tag)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
