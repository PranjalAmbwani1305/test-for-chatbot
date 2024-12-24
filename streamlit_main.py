import os
from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import login
from langchain.vectorstores import Pinecone  
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.llms import HuggingFaceEndpoint
from pinecone import Pinecone as PineconeClient, ServerlessSpec  

# Load environment variables
load_dotenv()

# HuggingFace API Login
login(token=os.getenv('HUGGINGFACE_API_TOKEN'))

# Set up Pinecone and HuggingFace API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Define the index name
index_name = "textembedding"

# Initialize Pinecone client
pc = PineconeClient(api_key=PINECONE_API_KEY)

# Create Pinecone index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # dimension of sentence embeddings
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        )
    )

# Load the PDF document
pdf_path = "gpmc.pdf"
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
repo_id = "sentence-transformers/all-MiniLM-L6-v2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, 
    temperature=0.8, 
    top_k=50, 
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_TOKEN')
)

# Chatbot class to handle queries
class Chatbot:
    def ask(self, question):
        # Retrieve relevant document chunks
        retriever = docsearch.as_retriever()
        relevant_docs = retriever.get_relevant_documents(question)  # Correct method here
        
        # Get the context from the relevant documents
        context = "\n".join([doc.page_content for doc in relevant_docs])  # Use 'page_content' to get the text
        
        # Generate the answer using the Hugging Face model
        response = llm.invoke({"context": context, "question": question})
        return response

# Set up the Streamlit UI
st.title("Chatbot")

# Get user input for the query
query = st.text_input("Enter your query:")

if query:
    with st.spinner("Generating response..."):
        try:
            chatbot = Chatbot()
            response = chatbot.ask(query)
            st.write("Answer:", response)
        except Exception as e:
            st.error(f"Error: {e}")
