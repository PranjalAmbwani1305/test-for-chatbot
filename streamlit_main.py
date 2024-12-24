import os
from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import login, hf_api
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.llms import HuggingFaceEndpoint
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Get the API tokens from environment variables
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Verify that the tokens are loaded properly
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("HuggingFace API Token is missing from the environment variables.")
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API Key is missing from the environment variables.")
if not PINECONE_ENV:
    raise ValueError("Pinecone Environment is missing from the environment variables.")
if not PINECONE_INDEX_NAME:
    raise ValueError("Pinecone Index Name is missing from the environment variables.")

# HuggingFace API Login
login(token=HUGGINGFACE_API_TOKEN)

# Verify HuggingFace token is working by getting user info
try:
    user_info = hf_api.whoami()
    print(f"HuggingFace Authenticated as: {user_info['name']}")
except Exception as e:
    raise ValueError(f"HuggingFace authentication failed: {str(e)}")

# Initialize Pinecone client
try:
    pc = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    print("Pinecone client initialized.")
except Exception as e:
    raise ValueError(f"Pinecone client initialization failed: {str(e)}")

# Create Pinecone index if it doesn't exist
index_name = PINECONE_INDEX_NAME or "textembedding"  # Default to 'textembedding' if not provided

try:
    existing_indexes = pc.list_indexes()
    print(f"Existing indexes: {existing_indexes}")
    
    if index_name not in existing_indexes:
        print(f"Index {index_name} does not exist. Creating a new one...")
        # Create a new Pinecone index
        pc.create_index(
            name=index_name,
            dimension=384,  # dimension of sentence embeddings (verify this is correct for your embeddings model)
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Index {index_name} created successfully.")
    else:
        print(f"Index {index_name} already exists.")
except Exception as e:
    print(f"Error creating Pinecone index: {str(e)}")  # Log the error to the console
    raise ValueError(f"Error creating Pinecone index: {str(e)}")  # Reraise the exception for Streamlit to capture it

# Load the PDF document
pdf_path = "gpmc.pdf"  # Replace this with your PDF file path
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

# Split documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=4)
docs = text_splitter.split_documents(documents)

# Use sentence-transformers/all-MiniLM-L6-v2 model for embeddings
repo_id = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=repo_id)

# Set up Pinecone for storing embeddings
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# Set up Hugging Face model
llm = HuggingFaceEndpoint(
    repo
