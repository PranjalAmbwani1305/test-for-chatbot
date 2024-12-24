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
index_name =  os.getenv("index_name")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings()

pinecone.create_index("textembedding", dimension=384)

doc_store = Pinecone.from_existing_index(
    index_name="textembedding",  # Replace with your actual Pinecone index name
    embedding_function=embeddings.embed_query  # Function to get the embedding
)
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
