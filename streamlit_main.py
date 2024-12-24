import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import pinecone
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "pdf-query-chatbot"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

doc_store = Pinecone(index_name, embeddings.embed_query, "text")

st.title("PDF Query Chatbot")
st.sidebar.header("Upload PDF")

uploaded_file = st.sidebar.file_uploader("Upload your PDF file", type=["pdf"])
if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    chunk_size = 500
    text_chunks = [pdf_text[i : i + chunk_size] for i in range(0, len(pdf_text), chunk_size)]

    with st.spinner("Indexing document..."):
        for chunk in text_chunks:
            doc_store.add_texts([chunk])

    st.success("Document indexed successfully!")
