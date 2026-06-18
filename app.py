import os
import re
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.retrievers import BM25Retriever

# ── API Keys ──────────────────────────────────────────────────────────────────
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# LangSmith tracing — optional (enable if key is present)
_langchain_key = st.secrets.get("LANGCHAIN_API_KEY", "")
if _langchain_key:
    os.environ["LANGCHAIN_API_KEY"]      = _langchain_key
    os.environ["LANGCHAIN_TRACING_V2"]   = "true"
    os.environ["LANGCHAIN_PROJECT"]      = "zyro-rag-challenge"
else:
    os.environ["LANGCHAIN_TRACING_V2"]   = "false"

# ── Standardized refusal message ──────────────────────────────────────────────
REFUSAL_MSG = (
    "I can only answer questions related to Zyro Dynamics HR policies. "
    "This question is outside my scope. Please contact HR directly for non-policy queries."
)

# ── HR keyword fast-path (avoid unnecessary LLM calls for clear HR questions) ─
HR_KEYWORDS = {
    "leave", "vacation", "pto", "sick", "maternity", "paternity", "wfh",
    "remote", "hybrid", "office", "salary", "ctc", "bonus", "increment",
    "appraisal", "performance", "pip", "review", "rating", "probation",
    "onboarding", "offboarding", "separation", "resignation", "notice",
    "travel", "expense", "reimbursement", "policy", "handbook", "conduct",
    "harassment", "posh", "icc", "device", "security", "it", "data",
    "confidential", "employee", "hr", "benefit", "insurance", "pf",
    "gratuity", "esic", "holiday", "overtime", "shift", "allowance",
    "deduction", "tax", "tds", "grade", "band", "promotion", "transfer",
    "zyro", "dynamics", "company", "joining", "full", "final", "settlement",
    "termination", "disciplinary", "code", "ethics", "misconduct", "warning",
    "annual", "earned", "casual", "compensatory", "bereavement",
    "accommodation", "hotel", "flight", "per diem", "cab", "conveyance",
    "work", "hours", "attendance", "payroll", "stipend", "internship",
    "retirement", "okr", "objectives", "feedback", "calibration",
    "encryption", "laptop", "password", "vpn", "firewall", "antivirus",
    "marriage", "wedding", "encashment", "carry", "forward",
    "probation", "confirmation", "induction", "relieving", "experience",
    "letter", "exit", "interview", "handover", "knowledge",
}

OUT_OF_SCOPE_KEYWORDS = {
    "weather", "cricket", "football", "sports", "movie", "recipe", "cook",
    "stock", "crypto", "bitcoin", "politics", "government", "news",
    "celebrity", "music", "song", "game", "gaming", "restaurant", "food",
    "travel destination", "tourist", "covid vaccine", "hospital", "doctor",
    "astrology", "horoscope", "joke", "poem", "story", "history of india",
    "medical diagnosis", "investment", "share market", "lottery", "betting",
    "capital", "president", "prime minister", "planet", "solar system",
    "programming", "python", "javascript", "machine learning", "ai model",
}


def is_hr_question(question: str):
    """Fast keyword-based HR relevance check."""
    q_lower = question.lower()
    words = set(re.findall(r'\b\w+\b', q_lower))
    # If clearly out-of-scope → False
    if words & OUT_OF_SCOPE_KEYWORDS:
        return False
    # If any HR keyword found → True
    if words & HR_KEYWORDS:
        return True
    return None  # Ambiguous → defer to LLM


# ── Pipeline (cached so Streamlit doesn't reload on every interaction) ─────────
@st.cache_resource(show_spinner="⚙️ Loading HR knowledge base…")
def load_pipeline():
    # 1. Load PDFs
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir    = os.path.join(current_dir, "docs")
    if not os.path.exists(docs_dir):
        docs_dir = current_dir

    loader    = PyPDFDirectoryLoader(docs_dir)
    documents = loader.load()
    if not documents:
        st.error("❌ No PDFs found in /docs folder!")
        st.stop()

    # ── Pre-extract FULL TEXT per source document ─────────────────────────────
    # This is the key to parent-document retrieval: when retrieval finds
    # relevant chunks, we inject the FULL document text as context so
    # no table rows, numbers, or details are ever lost to chunking.
    doc_full_texts = {}
    for doc in documents:
        source = doc.metadata.get("source", "")
        if source not in doc_full_texts:
            doc_full_texts[source] = ""
        doc_full_texts[source] += doc.page_content + "\n\n"

    # 2. Chunk — SMALLER chunks for finer-grained retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    # 3. Embed — BAAI/bge-base-en-v1.5
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 4. FAISS semantic retriever (MMR for diversity)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.5},
    )

    # 5. BM25 keyword retriever (catches exact term matches embeddings miss)
    bm25_retriever = BM25Retriever.from_documents(chunks, k=10)

    # 6. Hybrid ensemble retriever — best of both worlds
    # We will manually merge from both bm25_retriever and semantic_retriever

    # 7. LLM — Groq llama-3.3-70b-versatile
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=2048,
    )

    # 8. Multi-query prompt — generates query variants for better recall
    multi_query_prompt = ChatPromptTemplate.from_template(
        """Generate 3 different phrasings of the following HR question to improve document retrieval.
Output ONLY the 3 questions, one per line, no numbering, no extra text.

Original question: {question}

3 rephrased questions:"""
    )

    def multi_query_retrieve(question: str):
        """Generate query variants and retrieve docs for each, deduplicated."""
        try:
            variants_raw = (multi_query_prompt | llm | StrOutputParser()).invoke({"question": question})
            variants = [q.strip() for q in variants_raw.strip().split("\n") if q.strip()][:3]
        except Exception:
            variants = []
        all_queries = [question] + variants
        seen_ids, docs = set(), []
        for q in all_queries:
            try:
                retrieved_docs = bm25_retriever.invoke(q) + semantic_retriever.invoke(q)
                for doc in retrieved_docs:
                    doc_id = hash(doc.page_content[:200])
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        docs.append(doc)
            except Exception:
                pass
        return docs or (bm25_retriever.invoke(question) + semantic_retriever.invoke(question))

    def get_full_document_context(question: str):
        """Parent-document retrieval: find relevant chunks, then inject
        FULL text of the top matching source documents as context.
        This ensures tables, lists, and multi-part policies are never fragmented."""
        initial_docs = multi_query_retrieve(question)

        if not initial_docs:
            return "", []

        # Count which source documents appear most in retrieval results
        source_counts = {}
        for doc in initial_docs:
            src = doc.metadata.get("source", "")
            source_counts[src] = source_counts.get(src, 0) + 1

        # Get top 3 most-relevant source documents
        top_sources = sorted(source_counts, key=source_counts.get, reverse=True)[:3]

        # Build context from FULL document texts (no fragmentation!)
        context_parts = []
        for src in top_sources:
            if src in doc_full_texts:
                context_parts.append(doc_full_texts[src])

        return "\n\n---\n\n".join(context_parts), top_sources

    # 9. Master RAG prompt — PRECISION focused for maximum semantic similarity
    rag_prompt = ChatPromptTemplate.from_template("""You are the HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer the employee's question using ONLY the policy documents provided below.

INSTRUCTIONS:
1. Use ONLY information from the Context. Never add external knowledge.
2. Be PRECISE: state exact numbers, days, percentages, amounts, dates, and eligibility criteria directly from the policy.
3. Be COMPLETE: include ALL relevant details, conditions, exceptions, and related rules.
4. Be DIRECT: start with the answer immediately. Do NOT add filler phrases like "According to the policy" or "As per the HR documents" or "Based on the provided context".
5. For questions about entitlements or limits, state the specific values clearly (e.g., "8 days of Casual Leave per year").
6. For questions about processes, list the steps in order.
7. For tabular data (travel entitlements, leave types, rating scales), present the key data points clearly.
8. Use bullet points for multi-part answers.
9. If the context does NOT contain the answer, respond ONLY with: "This information is not available in the HR policy documents. Please contact HR directly."

Context (from Zyro Dynamics HR Policy Documents):
{context}

Employee Question: {question}

Answer:""")

    # 10. Scope-check prompt
    scope_prompt = ChatPromptTemplate.from_template("""You are a classifier. Determine if the following question is related to:
- Company HR policies, employment, workplace rules
- Leave, attendance, holidays
- Benefits, compensation, salary, insurance
- Workplace conduct, ethics, harassment
- IT security, data protection, devices
- Onboarding, separation, probation, retirement
- Travel expenses, reimbursements
- Performance management, reviews, promotions

Answer ONLY with one word: YES or NO.

Question: {question}
Answer:""")

    def ask_bot(question: str):
        question = question.strip()
        if not question:
            return {"answer": "Please ask a question.", "sources": []}

        # Layer 1: Keyword fast-path
        keyword_result = is_hr_question(question)

        if keyword_result is False:
            return {"answer": REFUSAL_MSG, "sources": []}

        # Layer 2: LLM classification for ambiguous questions
        if keyword_result is None:
            try:
                classification = (scope_prompt | llm | StrOutputParser()).invoke(
                    {"question": question}
                ).strip().upper()
                if "NO" in classification and "YES" not in classification:
                    return {"answer": REFUSAL_MSG, "sources": []}
            except Exception:
                # If classification fails, default to trying to answer
                pass

        # Layer 3: Parent-document context retrieval
        context, sources = get_full_document_context(question)

        if not context.strip():
            return {
                "answer": "This information is not available in the HR policy documents. Please contact HR directly.",
                "sources": []
            }

        # Layer 4: Generate precise answer with full document context
        chain = (
            {"context": RunnableLambda(lambda _: context), "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)

        return {"answer": answer, "sources": sources}

    return ask_bot


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Header */
.zyro-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.zyro-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.zyro-header p {
    color: #94a3b8;
    font-size: 0.95rem;
}

/* Chat messages */
.stChatMessage {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(8px) !important;
    margin-bottom: 0.8rem !important;
    padding: 1rem 1.2rem !important;
}

/* User message accent */
[data-testid="stChatMessage"][data-testid*="user"] {
    border-left: 3px solid #a78bfa !important;
}

/* Chat input */
.stChatInputContainer {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(167,139,250,0.4) !important;
    backdrop-filter: blur(8px) !important;
}

/* Markdown text in chat */
.stMarkdown p, .stMarkdown li {
    color: #e2e8f0 !important;
    line-height: 1.7 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

/* Source badge */
.source-badge {
    display: inline-block;
    background: rgba(167,139,250,0.2);
    border: 1px solid rgba(167,139,250,0.4);
    color: #c4b5fd;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    margin: 3px 3px;
    font-weight: 500;
}

/* Stats cards */
.stat-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
    backdrop-filter: blur(6px);
}
.stat-number {
    font-size: 1.5rem;
    font-weight: 700;
    color: #a78bfa;
}
.stat-label {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 2px;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #a78bfa !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 Zyro Dynamics")
    st.markdown("**HR Help Desk** — AI Powered")
    st.divider()

    st.markdown("### 📋 Policy Coverage")
    policies = [
        ("🏛️", "Company Profile"),
        ("📖", "Employee Handbook"),
        ("🌴", "Leave Policy"),
        ("🏠", "Work From Home"),
        ("⚖️", "Code of Conduct"),
        ("📊", "Performance Review"),
        ("💰", "Compensation & Benefits"),
        ("🔒", "IT & Data Security"),
        ("🛡️", "POSH Policy"),
        ("🚀", "Onboarding & Separation"),
        ("✈️", "Travel & Expense"),
    ]
    for icon, name in policies:
        st.markdown(f"<small>{icon} {name}</small>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "How many casual leaves do I get per year?",
        "What is the WFH policy for engineers?",
        "What are the travel reimbursement limits?",
        "How does the performance rating system work?",
    ]
    for q in sample_questions:
        st.markdown(f"<small>• {q}</small>", unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<small style='color:#64748b'>⚡ Powered by Groq LLaMA-3.3-70B<br>"
        "🔍 Hybrid BM25 + BGE Embeddings<br>"
        "📄 Parent-Document Retrieval<br>"
        "🔗 LangSmith Tracing Active</small>",
        unsafe_allow_html=True
    )

# ── Main Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="zyro-header">
    <h1>🤖 Zyro Dynamics HR Help Desk</h1>
    <p>Ask any HR policy question — powered by RAG with hybrid search across 11 policy documents</p>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="stat-number">11</div><div class="stat-label">Policy Docs</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="stat-number">Hybrid</div><div class="stat-label">BM25 + Semantic</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="stat-number">Parent</div><div class="stat-label">Doc Retrieval</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-card"><div class="stat-number">70B</div><div class="stat-label">LLaMA Model</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Load Pipeline ──────────────────────────────────────────────────────────────
ask_bot = load_pipeline()

# ── Chat State ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I'm the **Zyro Dynamics HR Assistant**. I can answer questions about our company's HR policies including leave, compensation, WFH, performance reviews, travel expenses, and more.\n\nHow can I help you today?",
        "sources": []
    })

# ── Render Chat History ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Source Documents", expanded=False):
                badges = ""
                for src in msg["sources"]:
                    name = os.path.basename(src).replace(".pdf", "").replace("_", " ")
                    # Remove leading number prefix (e.g., "00 ", "01 ")
                    name = re.sub(r"^\d+\s*", "", name).title()
                    badges += f'<span class="source-badge">📋 {name}</span>'
                st.markdown(badges, unsafe_allow_html=True)

# ── Chat Input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask an HR policy question…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate & show assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching HR policies…"):
            response = ask_bot(prompt)

        st.markdown(response["answer"])

        if response.get("sources"):
            with st.expander("📄 Source Documents", expanded=False):
                badges = ""
                for src in response["sources"]:
                    name = os.path.basename(src).replace(".pdf", "").replace("_", " ")
                    name = re.sub(r"^\d+\s*", "", name).title()
                    badges += f'<span class="source-badge">📋 {name}</span>'
                st.markdown(badges, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response.get("sources", [])
    })

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.75rem'>"
    "Zyro Dynamics HR Help Desk • RAG-Powered • For internal use only"
    "</div>",
    unsafe_allow_html=True
)
