# import os
# import streamlit as st
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# # Fetch Groq Key from Streamlit Secrets
# os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# @st.cache_resource
# def load_rag_pipeline():
#     # 1. Find the exact path where your app.py lives on Streamlit's servers
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     docs_dir = os.path.join(current_dir, "docs")
    
#     # Fallback: If 'docs' folder doesn't exist, look in the main folder
#     if not os.path.exists(docs_dir):
#         docs_dir = current_dir
        
#     # 2. Load the PDFs
#     loader = PyPDFDirectoryLoader(docs_dir)
#     documents = loader.load()
    
#     # 3. Guardrail: Stop the app and show a clear error if no PDFs are found
#     if len(documents) == 0:
#         st.error("❌ ERROR: No PDF files were found in your GitHub repository!")
#         st.info("👉 Please ensure you created a folder named 'docs' and uploaded the 11 extracted .pdf files inside it (not the .zip file).")
#         st.stop()
        
#     print(f"✅ Successfully loaded {len(documents)} documents.")
    
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_documents(documents)
    
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_store = FAISS.from_documents(chunks, embeddings)
#     retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
    
#     llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
#     template = """You are an AI HR assistant for Zyro Dynamics. Answer using ONLY the context.
# If OUT OF SCOPE or not in context, reply EXACTLY with: "I can only answer HR-related questions from Zyro Dynamics policy documents."
# Context: {context}
# Question: {question}
# Answer:"""
    
#     prompt = ChatPromptTemplate.from_template(template)
    
#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)
        
#     return (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt | llm | StrOutputParser()
#     )

# st.title("Zyro Dynamics HR Help Desk 🤖")

# # Load the chain
# chain = load_rag_pipeline()

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("Ask an HR question..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)
        
#     with st.chat_message("assistant"):
#         with st.spinner("Searching policies..."):
#             response = chain.invoke(prompt)
import os
import streamlit as st
import uuid
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

# Securely load Groq Key
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

class HRState(TypedDict):
    question: str
    context: str
    answer: str
    is_hr: bool

@st.cache_resource
def load_graph():
    # 1. Load Docs
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(current_dir, "docs")
    if not os.path.exists(docs_dir): docs_dir = current_dir
    
    loader = PyPDFDirectoryLoader(docs_dir)
    documents = loader.load()
    if len(documents) == 0:
        st.error("No PDFs found in /docs folder!")
        st.stop()
        
    # 2. Chunk & Embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 3. ChromaDB
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name="zyro_hr")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
    
    # 4. LLM (Using Groq for Cloud Deployment)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    # 5. LangGraph Nodes
    def node_classify(state: HRState):
        prompt = ChatPromptTemplate.from_template("Is this related to HR policies? YES or NO.\nQuestion: {question}\nAnswer:")
        res = (prompt | llm | StrOutputParser()).invoke({"question": state["question"]})
        return {"is_hr": "YES" in res.upper()}

    def node_retrieve(state: HRState):
        docs = retriever.invoke(state["question"])
        return {"context": "\n\n".join(doc.page_content for doc in docs)}

    def node_generate(state: HRState):
        prompt = ChatPromptTemplate.from_template("Answer using ONLY context. Be concise.\nContext: {context}\nQuestion: {question}\nAnswer:")
        ans = (prompt | llm | StrOutputParser()).invoke({"context": state["context"], "question": state["question"]})
        return {"answer": ans}

    def node_refuse(state: HRState):
        return {"answer": "I can only answer HR-related questions from Zyro Dynamics policy documents."}

    # 6. Build Graph
    graph = StateGraph(HRState)
    graph.add_node("classify", node_classify)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("generate", node_generate)
    graph.add_node("refuse", node_refuse)
    graph.set_entry_point("classify")
    
    graph.add_conditional_edges("classify", lambda s: "retrieve" if s["is_hr"] else "refuse", {"retrieve": "retrieve", "refuse": "refuse"})
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("refuse", END)
    
    return graph.compile(checkpointer=MemorySaver())

# --- STREAMLIT UI & SESSION MANAGEMENT ---
st.title("Zyro Dynamics HR Help Desk 🤖")
st.caption("Powered by LangGraph, ChromaDB, and Llama 3.3")

app = load_graph()

# Initialize Session Management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Consulting HR policies..."):
            # Pass the session_id to MemorySaver so it remembers the user!
            config = {"configurable": {"thread_id": st.session_state.session_id}}
            response = app.invoke({"question": prompt}, config)
            
        st.markdown(response["answer"])
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

# Sidebar for Session Management
with st.sidebar:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
#         st.markdown(response)
#         st.session_state.messages.append({"role": "assistant", "content": response})
