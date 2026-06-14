import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def load_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(current_dir, "docs")
    if not os.path.exists(docs_dir): 
        docs_dir = current_dir
    
    loader = PyPDFDirectoryLoader(docs_dir)
    documents = loader.load()
    if not documents:
        st.error("No PDFs found in /docs folder!")
        st.stop()
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=300, 
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Bulletproof FAISS + MMR (Guaranteed to work on Streamlit Cloud)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    rag_prompt = ChatPromptTemplate.from_template(
        "Answer using ONLY the provided context. Be extremely concise and direct. Do NOT use filler phrases.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt | llm | StrOutputParser()
    )
    
    def ask_bot(question: str):
        p = ChatPromptTemplate.from_template("Is this question related to company HR policies, leave, benefits, rules, or employee guidelines? Answer ONLY 'YES' or 'NO'.\n\nQuestion: {question}\n\nAnswer:")
        res = (p | llm | StrOutputParser()).invoke({"question": question}).strip().upper()
        
        if "NO" in res:
            return {
                "answer": "I can only answer HR-related questions from Zyro Dynamics policy documents.", 
                "sources": []
            }
            
        docs = retriever.invoke(question)
        ans = rag_chain.invoke(question)
        sources = list(set([doc.metadata.get('source', 'Unknown Policy') for doc in docs]))
        
        return {"answer": ans, "sources": sources}

    return ask_bot

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .stChatMessage { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    .stTextInput > div > div > input { background-color: #f0f2f6; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Zyro Dynamics HR Help Desk")
st.caption("Production-Grade RAG | MMR Retrieval & Strict Guardrails")

ask_bot = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 View Source Documents"):
                for src in msg["sources"]:
                    clean_name = os.path.basename(src).replace('.pdf', '').replace('_', ' ').title()
                    st.write(f"- {clean_name}")

if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Consulting HR policies..."):
            response = ask_bot(prompt)
            
        st.markdown(response["answer"])
        if response.get("sources"):
            with st.expander("📄 View Source Documents"):
                for src in response["sources"]:
                    clean_name = os.path.basename(src).replace('.pdf', '').replace('_', ' ').title()
                    st.write(f"- {clean_name}")
                    
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response["answer"],
            "sources": response.get("sources", [])
        })
