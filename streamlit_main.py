# Chatbot Query from Pre-defined PDF in Streamlit Cloud using Hugging Face and Pinecone Langchain

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
import pinecone

# Load API keys from secrets
HUGGINGFACE_API_KEY = st.secrets["huggingface_api_key"]
PINECONE_API_KEY = st.secrets["pinecone_api_key"]
PINECONE_ENV = st.secrets["pinecone_env"]

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Load PDF
pdf_loader = PyPDFLoader("path_to_your_pdf.pdf")
documents = pdf_loader.load()

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="your_huggingface_model", api_key=HUGGINGFACE_API_KEY)

# Create Pinecone index
index_name = "your_index_name"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=embeddings.embed_dimension)

# Create vector store
vector_store = Pinecone.from_documents(documents, embeddings, index_name=index_name)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=embeddings, chain_type="stuff", retriever=vector_store.as_retriever())

# Streamlit UI
st.title("Chatbot Query Interface")
user_query = st.text_input("Ask a question about the PDF:")
if user_query:
    response = qa_chain.run(user_query)
    st.write(response)
