import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone as PineconeClient
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import pinecone
import os
import io 

os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

pdf_loader = PyPDFLoader("gpmc.pdf")
documents = pdf_loader.load()

embeddings = HuggingFaceEmbeddings(model_name="dyumat/mistral-7b-chat-pdf")

index_name = "chatbot"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=embeddings.embed_dimension)

vector_store = Pinecone.from_documents(documents, embeddings, index_name=index_name)

llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=HUGGINGFACE_API_KEY)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

st.title("Chatbot Query Interface")
user_query = st.text_input("Ask a question about the PDF:")

if user_query:
    response = qa_chain.run(user_query)
    st.write(response)
