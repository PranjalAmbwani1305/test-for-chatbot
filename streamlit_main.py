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
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", api_key=HUGGINGFACE_API_TOKEN)

try:
    doc_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME)
    st.success("Document store created successfully!")

except Exception as e:
    st.error(f"Failed to create document store: {e}")
    try:
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine"
        )
        st.success(f"Index '{PINECONE_INDEX_NAME}' created successfully!")
    except Exception as create_error:
        st.error(f"Failed to create index: {create_error}")
    doc_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME)

st.title("Chatbot")
pdf_path = "gpmc.pdf"
pdf_reader = PdfReader(pdf_path)
pdf_text = ""
for page in pdf_reader.pages:
    pdf_text += page.extract_text()

chunk_size = 500
text_chunks = [pdf_text[i : i + chunk_size] for i in range(0, len(pdf_text), chunk_size)]

with st.spinner("Indexing document..."):
    for chunk in text_chunks:
        embedding = embeddings.embed_query(chunk)
        doc_store.add_texts([chunk], embeddings=[embedding])

st.success("Document indexed successfully!")

query = "example query"
try:
    query_embedding = embeddings.embed_query(query)
    results = doc_store.similarity_search(query, k=5)
    st.write(f"Search results: {results}")
    st.success("Query executed successfully!")
except Exception as query_error:
    st.error(f"Failed to execute query: {query_error}")
