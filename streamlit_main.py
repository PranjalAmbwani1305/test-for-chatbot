import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PDF_FILE_PATH = './your_document.pdf'  # Replace with your PDF file path

# Check if the API key is set
if not PINECONE_API_KEY:
    st.error("Pinecone API key is not set. Please check your .env file.")
    st.stop()

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name = "pdf-chatbot-index"

    # List existing indexes
    existing_indexes = pc.list_indexes().names()
    st.write("Existing indexes:", existing_indexes)

    # Create or reset the Pinecone index
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,  # Adjust dimension based on your embeddings
            metric='cosine',  # or 'euclidean', depending on your use case
            spec=ServerlessSpec(
                cloud='aws',  # Specify your cloud provider
                region='us-west-2'  # Specify your region
            )
        )
        st.success(f"Index '{index_name}' created successfully.")
    else:
        st.write(f"Index '{index_name}' already exists.")

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
try:
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectordb = LangchainPinecone.from_documents(texts, embeddings, index_name=index_name)
except Exception as e:
    st.error(f"Error creating embeddings or storing in Pinecone: {e}")
    st.stop()

# Set up the Streamlit app
st.title("PDF Chatbot")
user_input = st.text_input("Ask a question about the PDF:")

if user_input:
    with st.spinner("Searching for an answer..."):
        # Create a conversational retrieval chain
        try:
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=embeddings,
                retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
                return_source_documents=True
            )
            response = qa_chain({'question': user_input})
            st.write("Response:", response['answer'])
        except Exception as e:
            st.error(f"Error during query processing: {e}")
