import os
import re
import json
import bs4
import streamlit as st
import pickle
import time
import plotly.graph_objects as go
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# ── Page config 
st.set_page_config(
    page_title="Equity Research Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS 
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #0a0e1a; color: #e2e8f0; }

  [data-testid="stSidebar"] { background: #0d1220 !important; border-right: 1px solid #1e2a3a; }
  [data-testid="stSidebar"] .stTextInput > div > div > input {
    background: #111827; border: 1px solid #1e2a3a; border-radius: 8px;
    color: #e2e8f0; font-family: 'DM Mono', monospace; font-size: 0.82rem;
    padding: 0.5rem 0.75rem; transition: border-color 0.2s;
  }
  [data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: #3b82f6; box-shadow: 0 0 0 2px rgba(59,130,246,0.15);
  }
  [data-testid="stSidebar"] label {
    color: #94a3b8 !important; font-size: 0.75rem !important;
    font-weight: 500 !important; letter-spacing: 0.05em; text-transform: uppercase;
  }

  .eq-header { padding: 2.5rem 0 1.5rem 0; border-bottom: 1px solid #1e2a3a; margin-bottom: 2rem; }
  .eq-header h1 { font-size: 2rem; font-weight: 700; color: #f1f5f9; margin: 0; letter-spacing: -0.02em; }
  .eq-header p  { color: #64748b; font-size: 0.9rem; margin: 0.35rem 0 0 0; }
  .eq-badge {
    display: inline-block; background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.3); color: #3b82f6;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 0.25rem 0.6rem; border-radius: 4px; margin-bottom: 0.75rem;
  }

  .step-card {
    background: #111827; border: 1px solid #1e2a3a; border-radius: 12px;
    padding: 1.25rem 1.5rem; margin-bottom: 0.75rem;
    display: flex; align-items: center; gap: 1rem;
  }
  .step-card.active  { border-color: #3b82f6; background: rgba(59,130,246,0.06); }
  .step-card.done    { border-color: #22c55e; background: rgba(34,197,94,0.05); }
  .step-card.error   { border-color: #ef4444; background: rgba(239,68,68,0.05); }
  .step-card.pending { opacity: 0.45; }
  .step-icon  { font-size: 1.3rem; width: 2rem; text-align: center; }
  .step-label { font-size: 0.85rem; font-weight: 500; color: #cbd5e1; }
  .step-sub   { font-size: 0.75rem; color: #475569; margin-top: 0.1rem; }

  .query-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: #64748b; margin-bottom: 0.5rem; }
  .stTextInput > div > div > input {
    background: #111827 !important; border: 1px solid #1e2a3a !important;
    border-radius: 10px !important; color: #f1f5f9 !important;
    font-size: 0.95rem !important; padding: 0.8rem 1rem !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
  }
  .stTextInput > div > div > input::placeholder { color: #334155 !important; }

  .answer-card {
    background: #111827; border: 1px solid #1e2a3a; border-left: 3px solid #3b82f6;
    border-radius: 12px; padding: 1.5rem 1.75rem; margin-top: 1.25rem;
    color: #e2e8f0; font-size: 0.95rem; line-height: 1.7;
  }
  .analysis-card {
    background: #111827; border: 1px solid #1e2a3a; border-left: 3px solid #a855f7;
    border-radius: 12px; padding: 1.5rem 1.75rem; margin-top: 1.25rem;
    color: #e2e8f0; font-size: 0.95rem; line-height: 1.7;
  }
  .answer-header  { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase; color: #3b82f6; margin-bottom: 0.75rem; }
  .analysis-header{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase; color: #a855f7; margin-bottom: 0.75rem; }
  .mode-tag {
    display: inline-block; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 0.2rem 0.55rem; border-radius: 4px; margin-bottom: 0.6rem;
  }
  .mode-retrieval { background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
  .mode-analysis  { background: rgba(168,85,247,0.15); color: #a855f7; border: 1px solid rgba(168,85,247,0.3); }

  .source-chip {
    display: inline-block; background: #0f172a; border: 1px solid #1e2a3a;
    border-radius: 6px; padding: 0.3rem 0.65rem; font-family: 'DM Mono', monospace;
    font-size: 0.72rem; color: #94a3b8; margin: 0.25rem 0.25rem 0 0; word-break: break-all;
  }

  [data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    padding: 0.65rem 1rem !important; width: 100% !important; margin-top: 0.5rem;
  }
  hr { border-color: #1e2a3a !important; }
  .info-box { background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2); border-radius: 10px; padding: 1rem 1.25rem; color: #93c5fd; font-size: 0.85rem; margin-bottom: 1rem; }
  .warn-box  { background: rgba(234,179,8,0.08); border: 1px solid rgba(234,179,8,0.25); border-radius: 10px; padding: 1rem 1.25rem; color: #fde047; font-size: 0.85rem; margin-bottom: 1rem; line-height: 1.6; }
  .ready-box { background: rgba(34,197,94,0.07); border: 1px solid rgba(34,197,94,0.2); border-radius: 10px; padding: 0.9rem 1.25rem; color: #86efac; font-size: 0.85rem; margin-bottom: 1.25rem; }
  .sidebar-heading { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #475569; margin-bottom: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# ── Constants 
FILE_PATH = "faiss_store.pkl"

# ── LLM 
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        max_tokens=2048,
    )

# ── Analyst prompt 
ANALYST_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a senior equity research analyst at a top investment bank.
You have been given the following source material extracted from financial news articles:

--- SOURCE MATERIAL ---
{context}
--- END SOURCE MATERIAL ---

The user has asked: {question}

Instructions:
- Use the source material as your PRIMARY knowledge base. Ground your answer in it.
- You are NOT limited to only quoting the text. As an analyst, you MUST reason, interpret, infer, and synthesise.
- If the question asks for analysis (e.g. risks, outlook, comparison, valuation, recommendation), provide a structured analytical response.
- If the question is factual and directly answerable from the text, answer it precisely.
- If numerical data is present (prices, revenue, growth %), highlight and interpret it.
- If the question is about a concept or market dynamic not explicitly in the text but clearly relevant, you may draw on your general financial knowledge to enrich the answer — clearly label such additions as "[Analyst View]".
- Format your response clearly. Use bullet points or sections for analysis questions.
- End with a one-line "Analyst Summary" in bold.

Answer:"""
)

# ── Router: classify the query 
def classify_query(query: str, llm) -> str:
    """Returns 'chart', 'analysis', or 'retrieval'."""
    msg = llm.invoke([
        SystemMessage(content=(
            "You are a query classifier. Given a user question about stocks/finance, "
            "reply with exactly ONE word:\n"
            "- 'chart' if they want a visual chart, graph, or price trend\n"
            "- 'analysis' if they want reasoning, comparison, outlook, risks, recommendation, or synthesis\n"
            "- 'retrieval' if they want a direct factual answer from the articles\n"
            "Reply with only the one word, nothing else."
        )),
        HumanMessage(content=query),
    ])
    label = msg.content.strip().lower()
    if "chart" in label:
        return "chart"
    if "analysis" in label or "analys" in label:
        return "analysis"
    return "retrieval"


# ── Chart generator 
def generate_chart(query: str, context: str, llm):
    extraction_prompt = f"""You are a data extraction assistant.
From the financial text below, extract numerical data relevant to this request: "{query}"

SOURCE TEXT:
{context[:3000]}

Return ONLY a valid JSON object in this exact format (no markdown, no explanation):
{{
  "chart_type": "bar" | "line" | "pie",
  "title": "chart title",
  "x_label": "x axis label",
  "y_label": "y axis label",
  "series": [
    {{"name": "series name", "x": ["label1", "label2"], "y": [val1, val2]}}
  ]
}}

If there is insufficient numerical data, return: {{"error": "insufficient data"}}"""

    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    raw = response.content.strip()

    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    try:
        data = json.loads(raw)
    except Exception:
        return None, "Could not parse chart data from the articles."

    if "error" in data:
        return None, f"⚠️ {data['error']} — not enough numerical data in the articles to draw a chart."

    chart_type = data.get("chart_type", "bar")
    title      = data.get("title", "Chart")
    x_label    = data.get("x_label", "")
    y_label    = data.get("y_label", "")
    series     = data.get("series", [])

    layout = dict(
        title=dict(text=title, font=dict(color="#e2e8f0", size=16)),
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#94a3b8", family="Inter"),
        xaxis=dict(title=x_label, gridcolor="#1e2a3a", color="#94a3b8"),
        yaxis=dict(title=y_label, gridcolor="#1e2a3a", color="#94a3b8"),
        legend=dict(bgcolor="#0a0e1a", bordercolor="#1e2a3a"),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    COLORS = ["#3b82f6", "#a855f7", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4"]

    if chart_type == "line":
        fig = go.Figure(layout=layout)
        for i, s in enumerate(series):
            fig.add_trace(go.Scatter(
                x=s["x"], y=s["y"], name=s["name"],
                mode="lines+markers",
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                marker=dict(size=6),
            ))
    elif chart_type == "pie":
        s = series[0] if series else {"x": [], "y": []}
        fig = go.Figure(go.Pie(
            labels=s["x"], values=s["y"],
            marker=dict(colors=COLORS),
            textfont=dict(color="#e2e8f0"),
        ), layout=layout)
    else:  # bar
        fig = go.Figure(layout=layout)
        for i, s in enumerate(series):
            fig.add_trace(go.Bar(
                x=s["x"], y=s["y"], name=s["name"],
                marker_color=COLORS[i % len(COLORS)],
            ))

    return fig, None


# ── Sidebar 
with st.sidebar:
    st.markdown('<div class="sidebar-heading">📰 Source Articles</div>', unsafe_allow_html=True)
    st.markdown("Paste up to **5** news article URLs to analyse.")
    st.markdown("---")

    urls = []
    for i in range(5):
        url = st.text_input(f"URL {i + 1}", placeholder="https://...", key=f"url_{i}")
        urls.append(url)

    process_url_clicked = st.button("⚡ Process URLs")

    st.markdown("---")
    st.markdown(
        '<div style="color:#334155;font-size:0.72rem;line-height:1.8">'
        '🔵 <b style="color:#3b82f6">Blue</b> = direct retrieval<br>'
        '🟣 <b style="color:#a855f7">Purple</b> = analyst reasoning<br>'
        '📊 Charts auto-generated from data<br><br>'
        'Powered by Gemini 2.5 Flash · FAISS · HuggingFace'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Main header 
st.markdown("""
<div class="eq-header">
  <div class="eq-badge">AI Research Tool</div>
  <h1>📈 Equity Research Analyst</h1>
  <p>Feed it news articles. Ask anything — factual lookups, deep analysis, or charts.</p>
</div>
""", unsafe_allow_html=True)


# ── Helpers 
def render_step(label: str, sub: str, icon: str, state: str):
    st.markdown(
        f'<div class="step-card {state}">'
        f'  <div class="step-icon">{icon}</div>'
        f'  <div><div class="step-label">{label}</div>'
        f'       <div class="step-sub">{sub}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Process URLs 
if process_url_clicked:
    valid_urls = [u.strip() for u in urls if u.strip()]

    if not valid_urls:
        st.warning("⚠️  Please enter at least one URL before processing.")
    else:
        st.markdown(f'<div class="info-box">🔗 Processing <strong>{len(valid_urls)}</strong> URL(s)…</div>', unsafe_allow_html=True)
        steps_placeholder = st.empty()

        with steps_placeholder.container():
            render_step("Loading articles",      "Fetching content from URLs…",           "🌐", "active")
            render_step("Splitting into chunks", "Recursive character splitter",          "✂️",  "pending")
            render_step("Building vector store", "HuggingFace embeddings → FAISS index", "🧠", "pending")
            render_step("Ready",                 "Knowledge base saved to disk",          "✅", "pending")

        loader = WebBaseLoader(
            web_paths=valid_urls,
            bs_kwargs={"parse_only": bs4.SoupStrainer(
                name=["article", "main", "section", "div", "p", "h1", "h2", "h3"]
            )},
            requests_kwargs={"headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }},
        )
        data = loader.load()
        time.sleep(0.3)

        with steps_placeholder.container():
            render_step("Loading articles",      f"Done — fetched {len(data)} page(s)",  "🌐", "done")
            render_step("Splitting into chunks", "Splitting documents…",                  "✂️",  "active")
            render_step("Building vector store", "HuggingFace embeddings → FAISS index", "🧠", "pending")
            render_step("Ready",                 "Knowledge base saved to disk",          "✅", "pending")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, separators=["\n\n", "\n", ".", ","], chunk_overlap=70,
        )
        docs = text_splitter.split_documents(data)
        time.sleep(0.3)

        if not docs:
            with steps_placeholder.container():
                render_step("Loading articles",      "Pages fetched — but no text extracted", "🌐", "error")
                render_step("Splitting into chunks", "0 chunks — nothing to embed",           "✂️",  "error")
                render_step("Building vector store", "Skipped",                               "🧠", "pending")
                render_step("Ready",                 "Cannot build knowledge base",           "✅", "pending")
            st.markdown(
                '<div class="warn-box"><strong>⚠️ No text extracted.</strong><br>'
                'Pages may be behind a paywall, JS-rendered, or blocking scrapers.<br>'
                '💡 Try archive.ph mirror links instead.</div>',
                unsafe_allow_html=True,
            )
            st.stop()

        with steps_placeholder.container():
            render_step("Loading articles",      f"Done — fetched {len(data)} page(s)",          "🌐", "done")
            render_step("Splitting into chunks", f"Done — {len(docs)} chunks created",           "✂️",  "done")
            render_step("Building vector store", "Embedding chunks and indexing with FAISS…",    "🧠", "active")
            render_step("Ready",                 "Knowledge base saved to disk",                 "✅", "pending")

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, embedding_model)

        with open(FILE_PATH, "wb") as f:
            pickle.dump(vector_store, f)

        with steps_placeholder.container():
            render_step("Loading articles",      f"Done — fetched {len(data)} page(s)",          "🌐", "done")
            render_step("Splitting into chunks", f"Done — {len(docs)} chunks created",           "✂️",  "done")
            render_step("Building vector store", "Done — FAISS index built and saved",           "🧠", "done")
            render_step("Ready",                 "Knowledge base is ready. Ask your question ↓", "✅", "done")

        time.sleep(0.4)
        st.balloons()


# ── Query 
st.markdown("---")

if os.path.exists(FILE_PATH):
    st.markdown('<div class="ready-box">✅ Knowledge base loaded — ask a factual question, request analysis, or ask for a chart.</div>', unsafe_allow_html=True)

st.markdown('<div class="query-label">Ask your question</div>', unsafe_allow_html=True)
user_query = st.text_input(
    label="Question",
    placeholder="e.g. What are the key risks? / Compare revenue growth / Show me a chart of margins",
    label_visibility="collapsed",
)

if user_query:
    if not os.path.exists(FILE_PATH):
        st.error("❌ No knowledge base found. Please process some URLs first.")
    else:
        llm = get_llm()

        with st.spinner("Routing query…"):
            mode = classify_query(user_query, llm)

        # ── Retrieve context 
        with open(FILE_PATH, "rb") as f:
            vector_store_loaded = pickle.load(f)

        retriever   = vector_store_loaded.as_retriever(search_kwargs={"k": 6})
        source_docs = retriever.invoke(user_query)
        context     = "\n\n".join(doc.page_content for doc in source_docs)

        # ── CHART mode 
        if mode == "chart":
            st.markdown(
                '<span class="mode-tag mode-analysis">📊 Chart Mode</span>',
                unsafe_allow_html=True,
            )
            with st.spinner("Extracting data and building chart…"):
                fig, err = generate_chart(user_query, context, llm)

            if err:
                st.warning(err)
                # Fall through to analysis if chart fails
                mode = "analysis"
            else:
                st.plotly_chart(fig, use_container_width=True)

                # Also give a brief text interpretation
                interp = llm.invoke([
                    SystemMessage(content="You are a senior equity analyst. Give a 2-3 sentence interpretation of the chart data below. Be concise and insightful."),
                    HumanMessage(content=f"Chart: {user_query}\nContext:\n{context[:1500]}"),
                ])
                st.markdown(
                    f'<div class="analysis-card">'
                    f'<div class="analysis-header">📝 Analyst Interpretation</div>'
                    f'{interp.content}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── ANALYSIS mode 
        if mode == "analysis":
            st.markdown(
                '<span class="mode-tag mode-analysis">🧠 Analyst Reasoning Mode</span>',
                unsafe_allow_html=True,
            )
            with st.spinner("Analysing…"):
                prompt_text = ANALYST_PROMPT.format(context=context, question=user_query)
                response = llm.invoke([HumanMessage(content=prompt_text)])
                answer   = response.content

            st.markdown(
                f'<div class="analysis-card">'
                f'<div class="analysis-header">🧠 Analyst Response</div>'
                f'{answer.replace(chr(10), "<br>")}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── RETRIEVAL mode 
        elif mode == "retrieval":
            st.markdown(
                '<span class="mode-tag mode-retrieval">🔍 Direct Retrieval Mode</span>',
                unsafe_allow_html=True,
            )
            with st.spinner("Retrieving answer…"):
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": ANALYST_PROMPT},
                )
                result = chain.invoke({"query": user_query})
                answer = result.get("result", "No answer returned.")

            st.markdown(
                f'<div class="answer-card">'
                f'<div class="answer-header">🔍 Answer</div>'
                f'{answer.replace(chr(10), "<br>")}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Sources 
        if source_docs:
            unique_sources = list({
                doc.metadata.get("source", "")
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