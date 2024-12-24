import os
import streamlit as st
import pinecone
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import time
import logging

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INDEX_NAME = "chatbot"
MODEL_NAME = "hkunlp/instructor-xl"
LLM_MODEL = "google/flan-t5-xxl"
CHUNK_SIZE = 512
TEMPERATURE = 0.5
MAX_LENGTH = 512
PDF_FILE_PATH = "gmpc.pdf"  # Path to your PDF

def initialize_resources():
    try:
        embeddings_model = HuggingFaceInstructEmbeddings(model_name=MODEL_NAME)
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
        llm = HuggingFaceHub(repo_id=LLM_MODEL, model_kwargs={"temperature": TEMPERATURE, "max_length": MAX_LENGTH})
        return embeddings_model, llm
    except Exception as e:
        logger.error(f"Error initializing resources: {e}")
        st.error(f"Error initializing resources: {e}")
        return None, None

def initialize_pinecone_index(embeddings_model):
    try:
        if INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(INDEX_NAME, dimension=len(embeddings_model.embed("test")))
        index = pinecone.Index(INDEX_NAME)
        return Pinecone(index, embeddings_model.embed_query)
    except Exception as e:
        logger.error(f"Pinecone index error: {e}")
        st.error(f"Pinecone index error: {e}")
        return None

def process_pdf(pdf_file_path, embeddings_model, vectorstore):
    try:
        loader = PyMuPDFLoader(pdf_file_path)
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        ids = [str(i) for i in range(len(chunks))]
        embeddings = [embeddings_model.embed(chunk) for chunk in chunks]
        if vectorstore:
            vectorstore.add_embeddings(text_embeddings=list(zip(embeddings, [{"text": chunk} for chunk in chunks])), ids=ids)
        logger.info(f"Processed {len(chunks)} from PDF.")
        return True
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        st.error(f"PDF processing error: {e}")
        return False

def main():
    st.title("GMP PDF Chatbot")

    embeddings_model, llm = initialize_resources()
    if not embeddings_model or not llm:
        return

    vectorstore = initialize_pinecone_index(embeddings_model)
    if not vectorstore:
        return

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state.chat_history = []

    # Process the PDF automatically if it exists
    if os.path.exists(PDF_FILE_PATH):
        with st.spinner("Processing PDF..."):
            if process_pdf(PDF_FILE_PATH, embeddings_model, vectorstore):
                memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=llm, retriever=vectorstore.as_retriever(), memory=memory
                )
                st.success("PDF processed. You can now ask questions.")
            else:
                st.error(f"Could not process {PDF_FILE_PATH}. Please ensure it exists.")
    else:
        st.error(f"PDF file '{PDF_FILE_PATH}' not found. Please place it in the same directory as the script.")
        return #stop the execution

    user_question = st.text_input("Ask a question:")

    if st.session_state.conversation and user_question:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversation({'question': user_question})
                st.session_state.chat_history = response['chat_history']
                for i, message in enumerate(st.session_state.chat_history):
                    speaker = "You" if i % 2 == 0 else "Bot"
                    st.markdown(f"**{speaker}:** {message.content}")
            except Exception as e:
                logger.error(f"Conversation error: {e}")
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
