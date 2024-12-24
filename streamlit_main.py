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
PINECONE_INDEX_NAME =  os.getenv("PINECONE_INDEX_NAME")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings()

try:
    doc_store = Pinecone(
        index_name=PINECONE_INDEX_NAME,  
        embedding_function=embeddings.embed_query 
    )
    st.success("Document store created successfully!")
except Exception as e:
    st.error(f"Failed to create document store: {e}")
    doc_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME)

    query = "example query"
    query_embedding = embeddings.embed_query(query)

    results = doc_store.similarity_search(query, k=5)  
    st.write(f"Search results: {results}")
    
    st.success("Document store created and query executed successfully!")

except Exception as e:
    st.error(f"Failed to create document store: {e}")
    
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
        doc_store.add_texts([chunk])

st.success("Document indexed successfully!")
