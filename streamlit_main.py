import os
import streamlit as st
import pinecone
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec  
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyMuPDFLoader  
from langchain.llms import HuggingFaceHub
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

    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    try:
        # Check if the index exists, if not, create it
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=len(embeddings[0]))

        # Connect to the index
        pinecone_index = pinecone.Index(index_name)

        # Prepare IDs for the embeddings
        ids = [str(i) for i in range(len(text_chunks))]
        
        # Upsert the embeddings into the Pinecone index
        pinecone_index.upsert(vectors=zip(ids, embeddings))

        logger.info(f"Uploaded {len(ids)} embeddings to Pinecone.")  # Log the success
        return Pinecone(index=pinecone_index, embedding_function=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl"))

    except Exception as e:
        logger.error(f"An error occurred while initializing Pinecone: {e}")
        return None

def load_pdf_and_process(file_path):
    """
    Load a PDF file and process it into text chunks.
    """
    logger.info(f"Loading PDF from {file_path}")
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    text_chunks = [doc.page_content for doc in documents]
    logger.info(f"Loaded {len(text_chunks)} chunks from the PDF.")
    return text_chunks

def main():
    st.title("PDF Embedding and Retrieval Chatbot")

    # Load the specific PDF file
    pdf_file_path = "gpmc.pdf"  # Ensure this file is in the same directory as your script

    if os.path.exists(pdf_file_path):
        # Load and process the PDF
        text_chunks = load_pdf_and_process(pdf_file_path)
        embeddings = generate_embeddings_for_chunks(text_chunks)

        # Initialize Pinecone and upload embeddings
        vector_store = initialize_pinecone_vector_store(text_chunks, embeddings)

        if vector_store is not None:
            st.success("Embeddings uploaded successfully!")

            # Set up conversation memory
            memory = ConversationBufferMemory()

            # Create a conversational retrieval chain
            llm = HuggingFaceHub(repo_id="gpt2")  # Replace with your desired model
            conversational_chain = ConversationalRetrievalChain(
                retriever=vector_store.as_retriever(),
                llm=llm,
                memory=memory
            )

            user_input = st.text_input("Ask a question:")
            if user_input:
                response = conversational_chain({"question": user_input})
                st.write(response['answer'])
        else:
            st.error("Failed to initialize Pinecone vector store.")
    else:
        st.error(f"The file {pdf_file_path} does not exist. Please ensure it is in the same directory.")

if __name__ == "__main__":
    main()
