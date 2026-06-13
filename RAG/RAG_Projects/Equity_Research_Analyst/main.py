import os
import streamlit as st
import pickle
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equity Research Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base & Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── App background ── */
  .stApp {
    background: #0a0e1a;
    color: #e2e8f0;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1e2a3a;
  }
  [data-testid="stSidebar"] .stTextInput > div > div > input {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    color: #e2e8f0;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    padding: 0.5rem 0.75rem;
    transition: border-color 0.2s;
  }
  [data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15);
  }
  [data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* ── Main header ── */
  .eq-header {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #1e2a3a;
    margin-bottom: 2rem;
  }
  .eq-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
    letter-spacing: -0.02em;
  }
  .eq-header p {
    color: #64748b;
    font-size: 0.9rem;
    margin: 0.35rem 0 0 0;
  }
  .eq-badge {
    display: inline-block;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.3);
    color: #3b82f6;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    margin-bottom: 0.75rem;
  }

  /* ── Step cards (progress) ── */
  .step-card {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color 0.3s, background 0.3s;
  }
  .step-card.active  { border-color: #3b82f6; background: rgba(59,130,246,0.06); }
  .step-card.done    { border-color: #22c55e; background: rgba(34,197,94,0.05); }
  .step-card.pending { opacity: 0.45; }
  .step-icon { font-size: 1.3rem; width: 2rem; text-align: center; }
  .step-label { font-size: 0.85rem; font-weight: 500; color: #cbd5e1; }
  .step-sub   { font-size: 0.75rem; color: #475569; margin-top: 0.1rem; }

  /* ── Query box ── */
  .query-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.5rem;
  }
  .stTextInput > div > div > input {
    background: #111827 !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-size: 0.95rem !important;
    padding: 0.8rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
  }
  .stTextInput > div > div > input::placeholder { color: #334155 !important; }

  /* ── Answer card ── */
  .answer-card {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-left: 3px solid #3b82f6;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-top: 1.25rem;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.7;
  }
  .answer-header {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 0.75rem;
  }

  /* ── Sources ── */
  .source-chip {
    display: inline-block;
    background: #0f172a;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 0.3rem 0.65rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #94a3b8;
    margin: 0.25rem 0.25rem 0 0;
    word-break: break-all;
  }
  .source-chip:hover { border-color: #3b82f6; color: #60a5fa; }

  /* ── Sidebar button ── */
  [data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.03em;
    padding: 0.65rem 1rem !important;
    width: 100% !important;
    margin-top: 0.5rem;
    transition: opacity 0.2s, transform 0.1s;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
  }

  /* ── Divider ── */
  hr { border-color: #1e2a3a !important; }

  /* ── Status / info boxes ── */
  .info-box {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: #93c5fd;
    font-size: 0.85rem;
    margin-bottom: 1rem;
  }
  .ready-box {
    background: rgba(34,197,94,0.07);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 10px;
    padding: 0.9rem 1.25rem;
    color: #86efac;
    font-size: 0.85rem;
    margin-bottom: 1.25rem;
  }

  /* ── Sidebar section heading ── */
  .sidebar-heading {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.75rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-heading">📰 Source Articles</div>', unsafe_allow_html=True)
    st.markdown("Paste up to **5** news article URLs to analyse.", unsafe_allow_html=False)
    st.markdown("---")

    urls = []
    for i in range(5):
        url = st.text_input(f"URL {i + 1}", placeholder="https://...", key=f"url_{i}", label_visibility="visible")
        urls.append(url)

    process_url_clicked = st.button("⚡ Process URLs")

    st.markdown("---")
    st.markdown(
        '<div style="color:#334155;font-size:0.72rem;line-height:1.6">'
        'Powered by Gemini 2.5 Flash · FAISS · HuggingFace Embeddings'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Main header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="eq-header">
  <div class="eq-badge">AI Research Tool</div>
  <h1>📈 Equity Research Analyst</h1>
  <p>Feed it news articles. Ask anything. Get grounded, source-backed answers.</p>
</div>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
FILE_PATH = "faiss_store.pkl"

def render_step(label: str, sub: str, icon: str, state: str):
    """state: 'pending' | 'active' | 'done'"""
    st.markdown(
        f'<div class="step-card {state}">'
        f'  <div class="step-icon">{icon}</div>'
        f'  <div><div class="step-label">{label}</div>'
        f'       <div class="step-sub">{sub}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Process URLs ───────────────────────────────────────────────────────────────
if process_url_clicked:
    valid_urls = [u.strip() for u in urls if u.strip()]

    if not valid_urls:
        st.warning("⚠️  Please enter at least one URL before processing.")
    else:
        st.markdown(f'<div class="info-box">🔗 Processing <strong>{len(valid_urls)}</strong> URL(s)…</div>', unsafe_allow_html=True)

        steps_placeholder = st.empty()

        # ── Step 1: Load ──
        with steps_placeholder.container():
            render_step("Loading articles",     "Fetching content from URLs",            "🌐", "active")
            render_step("Splitting into chunks", "Recursive character splitter",          "✂️",  "pending")
            render_step("Building vector store", "HuggingFace embeddings → FAISS index", "🧠", "pending")
            render_step("Ready",                 "Knowledge base saved to disk",          "✅", "pending")

        loader = UnstructuredURLLoader(urls=valid_urls)
        data = loader.load()
        time.sleep(0.3)

        # ── Step 2: Split ──
        with steps_placeholder.container():
            render_step("Loading articles",      "Done — loaded content from URLs",       "🌐", "done")
            render_step("Splitting into chunks", "Splitting documents…",                  "✂️",  "active")
            render_step("Building vector store", "HuggingFace embeddings → FAISS index", "🧠", "pending")
            render_step("Ready",                 "Knowledge base saved to disk",          "✅", "pending")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            separators=["\n\n", "\n", ".", ","],
            chunk_overlap=70,
        )
        docs = text_splitter.split_documents(data)
        time.sleep(0.3)

        # ── Step 3: Embed ──
        with steps_placeholder.container():
            render_step("Loading articles",      "Done — loaded content from URLs",           "🌐", "done")
            render_step("Splitting into chunks", f"Done — {len(docs)} chunks created",        "✂️",  "done")
            render_step("Building vector store", "Embedding chunks and indexing with FAISS…", "🧠", "active")
            render_step("Ready",                 "Knowledge base saved to disk",              "✅", "pending")

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, embedding_model)

        with open(FILE_PATH, "wb") as f:
            pickle.dump(vector_store, f)

        # ── All done ──
        with steps_placeholder.container():
            render_step("Loading articles",      "Done — loaded content from URLs",              "🌐", "done")
            render_step("Splitting into chunks", f"Done — {len(docs)} chunks created",           "✂️",  "done")
            render_step("Building vector store", "Done — FAISS index built and saved",           "🧠", "done")
            render_step("Ready",                 "Knowledge base is ready. Ask your question ↓", "✅", "done")

        time.sleep(0.4)
        st.balloons()


# ── Query ──────────────────────────────────────────────────────────────────────
st.markdown("---")

if os.path.exists(FILE_PATH):
    st.markdown('<div class="ready-box">✅ Knowledge base loaded — ready for questions.</div>', unsafe_allow_html=True)

st.markdown('<div class="query-label">Ask a question about the articles</div>', unsafe_allow_html=True)
user_query = st.text_input(
    label="Question",
    placeholder="e.g. What are the key risks mentioned in these articles?",
    label_visibility="collapsed",
)

if user_query:
    if not os.path.exists(FILE_PATH):
        st.error("❌ No knowledge base found. Please process some URLs first.")
    else:
        with st.spinner("Thinking…"):
            with open(FILE_PATH, "rb") as f:
                vector_store_loaded = pickle.load(f)

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.9,
                max_tokens=500,
            )

            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store_loaded.as_retriever(),
                return_source_documents=True,
            )

            result = chain.invoke({"query": user_query})

        # ── Answer ──
        answer = result.get("result", "No answer returned.")
        source_docs = result.get("source_documents", [])

        st.markdown(
            f'<div class="answer-card">'
            f'  <div class="answer-header">Answer</div>'
            f'  {answer}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Sources ──
        if source_docs:
            unique_sources = list({
                doc.metadata.get("source", "Unknown source")
                for doc in source_docs
                if doc.metadata.get("source")
            })

            if unique_sources:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.09em;'
                    'text-transform:uppercase;color:#475569;margin-bottom:0.5rem;">Sources</div>',
                    unsafe_allow_html=True,
                )
                chips = "".join(f'<span class="source-chip">🔗 {src}</span>' for src in unique_sources)
                st.markdown(chips, unsafe_allow_html=True)