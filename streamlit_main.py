import os
import streamlit as st
import pinecone
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import time
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_embeddings_for_chunks(text_chunks):
    """
    Generate embeddings for each chunk of text using HuggingFace model.
    """
    embeddings_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = [embeddings_model.embed(chunk) for chunk in text_chunks]
    logger.info(f"Generated {len(embeddings)} embeddings.")  # Log the number of embeddings
    return embeddings

def initialize_pinecone_vector_store(text_chunks, embeddings):
    """
    Initialize the Pinecone vector store and upload embeddings.
    """
    index_name = "chatbot"
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENV")

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    pinecone_client = pinecone

    try:
        if index_name not in pinecone_client.list_indexes():
            pinecone_client.create_index(index_name, dimension=len(embeddings[0]))

        pinecone_index = pinecone_client.Index(index_name)
        ids = [str(i) for i in range(len(text_chunks))]
        pinecone_index.upsert(vectors=zip(ids, embeddings))

        logger.info(f"Uploaded {len(ids)} embeddings to Pinecone.")  # Log the success
        return Pinecone(index=pinecone
