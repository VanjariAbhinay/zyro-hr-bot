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
}

OUT_OF_SCOPE_KEYWORDS = {
    "weather", "cricket", "football", "sports", "movie", "recipe", "cook",
    "stock", "crypto", "bitcoin", "politics", "government", "news",
    "celebrity", "music", "song", "game", "gaming", "restaurant", "food",
    "travel destination", "tourist", "covid vaccine", "hospital", "doctor",
    "astrology", "horoscope", "joke", "poem", "story", "history of india",
}

def is_hr_question(question: str) -> bool:
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

    # 2. Chunk — smaller chunks = more precise retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    # 3. Embed — BAAI/bge-base-en-v1.5: state-of-the-art retrieval, fits Streamlit Cloud
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # required for BGE cosine sim
    )

    # 4. FAISS vector store with MMR for diversity
    vectorstore = FAISS.from_documents(chunks, embeddings)
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.6},
    )

    # 5. LLM — Groq llama-3.3-70b-versatile (best free model)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
    )

    # 6. Multi-query prompt — generates query variants for better recall
    multi_query_prompt = ChatPromptTemplate.from_template(
        """Generate 3 different phrasings of the following HR question to improve document retrieval.
        Output ONLY the 3 questions, one per line, no numbering, no extra text.

        Original question: {question}

        3 rephrased questions:"""
    )

    def multi_query_retrieve(question: str, retriever, llm):
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
                for doc in retriever.invoke(q):
                    doc_id = hash(doc.page_content[:100])
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        docs.append(doc)
            except Exception:
                pass
        return docs or retriever.invoke(question)

    # 7. Master RAG prompt — structured for precise policy extraction
    rag_prompt = ChatPromptTemplate.from_template("""You are the official HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Your job is to answer employee HR questions STRICTLY using the provided policy documents.

CRITICAL RULES:
1. Use ONLY information from the Context below — never invent, assume, or add external knowledge.
2. Be SPECIFIC: include exact numbers, durations, percentages, eligibility criteria, and procedures.
3. If the context contains the answer, give it clearly and completely.
4. If the context does NOT contain sufficient information, say: "This specific information is not covered in the available HR policy documents. Please contact HR directly."
5. Do NOT hedge unnecessarily if the answer is clearly present in context.
6. Write in clear, professional English. Use bullet points for multi-part answers.

Context (from Zyro Dynamics HR Policy Documents):
{context}

Employee Question: {question}

HR Assistant Answer:""")

    # 8. Scope-check prompt
    scope_prompt = ChatPromptTemplate.from_template("""You are a classifier. Determine if the following question is related to company HR policies, employment, leave, benefits, compensation, workplace conduct, IT security, onboarding, travel expenses, or performance management at a company.

Answer ONLY with one word: YES or NO.

Question: {question}
Answer:""")

    def format_docs(docs):
        seen = set()
        result = []
        for doc in docs:
            content = doc.page_content.strip()
            if content not in seen:
                seen.add(content)
                result.append(content)
        return "\n\n---\n\n".join(result)

    def ask_bot(question: str):
        question = question.strip()
        if not question:
            return {"answer": "Please ask a question.", "sources": []}

        # Layer 1: Keyword fast-path
        keyword_result = is_hr_question(question)

        if keyword_result is False:
            return {
                "answer": "I'm sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents. This question appears to be outside my scope. Please contact HR directly for non-policy queries.",
                "sources": []
            }

        # Layer 2: LLM classification for ambiguous questions
        if keyword_result is None:
            classification = (scope_prompt | llm | StrOutputParser()).invoke(
                {"question": question}
            ).strip().upper()
            if "NO" in classification and "YES" not in classification:
                return {
                    "answer": "I'm sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents. This question appears to be outside my scope. Please contact HR directly for non-policy queries.",
                    "sources": []
                }

        # Layer 3: Retrieve with multi-query retrieval for better coverage
        docs = multi_query_retrieve(question, base_retriever, llm)

        if not docs:
            return {
                "answer": "I'm sorry, I couldn't find relevant information in the HR policy documents. Please contact HR directly.",
                "sources": []
            }

        context = format_docs(docs)

        # Layer 4: Check context relevance before generating answer
        chain = (
            {"context": RunnableLambda(lambda _: context), "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)

        # Collect unique source document names
        sources = sorted(set(
            doc.metadata.get("source", "Unknown") for doc in docs
        ))

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
        "🔍 BGE-Large Embeddings + MMR<br>"
        "🔗 LangSmith Tracing Active</small>",
        unsafe_allow_html=True
    )

# ── Main Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="zyro-header">
    <h1>🤖 Zyro Dynamics HR Help Desk</h1>
    <p>Ask any HR policy question — powered by RAG with semantic search across 11 policy documents</p>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="stat-number">11</div><div class="stat-label">Policy Docs</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="stat-number">BGE-L</div><div class="stat-label">Embedding Model</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="stat-number">MMR</div><div class="stat-label">Retrieval Strategy</div></div>', unsafe_allow_html=True)
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
