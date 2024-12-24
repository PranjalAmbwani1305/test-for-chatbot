import os
import streamlit as st
from pinecone import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PDF_FILE_PATH = 'gpmc.pdf'  # Replace with your PDF file path

if not PINECONE_API_KEY:
    st.error("Pinecone API key is not set. Please check your .env file.")
    st.stop()

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "pdf-chatbot-index"

    # List existing indexes
    existing_indexes = pc.list_indexes()
    st.write("Existing indexes:", existing_indexes)

    # Create or reset the Pinecone index
    if index_name in existing_indexes:
        pc.delete_index(index_name)
    pc.create_index(name=index_name, dimension=768, metric="cosine")

except Exception as e:
    st.error(f"Error initializing Pinecone: {e}")
    st.stop()

# Load PDF document
try:
    loader = PyPDFLoader(PDF_FILE_PATH)
    documents = loader.load()
except Exception as e:
    st.error(f"Error loading PDF: {e}")
    st.stop()

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Create embeddings and store in Pinecone
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)

# Set up the Streamlit app
st.title("PDF Chatbot")
user_input = st.text_input("Ask a question about the PDF:")

if user_input:
    with st.spinner("Searching for an answer..."):
        # Create a conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=embeddings,
            retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True
        )
        response = qa_chain({'question': user_input})
        st.write("Response:", response['answer'])
