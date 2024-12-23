import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import Pinecone  
from pinecone import Pinecone as PineconeClient, ServerlessSpec  
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Log in to Hugging Face using the secret key
login(token=st.secrets["general"]["HUGGINGFACE_API_KEY"])

class Chatbot:
    def __init__(self):
        # Check if the PDF file exists
        if not os.path.exists('gpmc.pdf'):
            st.error("The PDF file 'gpmc.pdf' was not found.")
            st.stop()

        # Load PDF data
        loader = PyMuPDFLoader('gpmc.pdf') 
        documents = loader.load()
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings()

        # Define the index name
        self.index_name = "amcgpmc"

        # Initialize Pinecone client using the secret key
        self.pc = PineconeClient(api_key=st.secrets["general"]["PINECONE_API_KEY"]) 
        # Create Pinecone index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'  
                )
            )

        # Set up Hugging Face model
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id, 
            temperature=0.8, 
            top_k=50, 
            huggingfacehub_api_token=st.secrets["general"]["HUGGINGFACE_API_KEY"]
        )

        # Define prompt template
