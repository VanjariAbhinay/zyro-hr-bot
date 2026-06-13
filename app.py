import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Fetch Groq Key from Streamlit Secrets
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def load_rag_pipeline():
    # 1. Find the exact path where your app.py lives on Streamlit's servers
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(current_dir, "docs")
    
    # Fallback: If 'docs' folder doesn't exist, look in the main folder
    if not os.path.exists(docs_dir):
        docs_dir = current_dir
        
    # 2. Load the PDFs
    loader = PyPDFDirectoryLoader(docs_dir)
    documents = loader.load()
    
    # 3. Guardrail: Stop the app and show a clear error if no PDFs are found
    if len(documents) == 0:
        st.error("❌ ERROR: No PDF files were found in your GitHub repository!")
        st.info("👉 Please ensure you created a folder named 'docs' and uploaded the 11 extracted .pdf files inside it (not the .zip file).")
        st.stop()
        
    print(f"✅ Successfully loaded {len(documents)} documents.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    template = """You are an AI HR assistant for Zyro Dynamics. Answer using ONLY the context.
If OUT OF SCOPE or not in context, reply EXACTLY with: "I can only answer HR-related questions from Zyro Dynamics policy documents."
Context: {context}
Question: {question}
Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

st.title("Zyro Dynamics HR Help Desk 🤖")

# Load the chain
chain = load_rag_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            response = chain.invoke(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
