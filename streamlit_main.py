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
        return Pinecone(index=pinecone_index, embedding_function=generate_embeddings_for_chunks)
    except Exception as e:
        st.error(f"Error initializing Pinecone vector store: {e}")
        logger.error(f"Error initializing Pinecone vector store: {e}")  # Log the error
        return None

def create_conversational_chain(vectorstore):
    """
    Create a conversational chain using HuggingFace LLM and vector store retriever.
    """
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    logger.info("Creating conversation chain...")  # Log chain creation
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def handle_conversation(user_question):
    """
    Handle user queries using the conversational chain and simulate a bot response.
    """
    if st.session_state.conversation:
        try:
            with st.spinner("Bot is typing..."):
                time.sleep(2)  # Simulate typing time
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            # Show the conversation history with better formatting
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:  # User's message
                    st.markdown(f"**You**: {message.content}")
                else:  # Bot's message
                    st.markdown(f"**Bot**: {message.content}")
        except Exception as e:
            st.error(f"An error occurred during conversation handling: {e}")
            logger.error(f"An error occurred during conversation handling: {e}")  # Log the error
    else:
        st.warning("The conversation has not been initialized yet. Please ensure the PDF is processed correctly.")

def extract_text_from_pdf(pdf_file_path):
    """
    Extract text from the given PDF file using PyMuPDF.
    """
    try:
        loader = PyMuPDFLoader(pdf_file_path)
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])  # Assuming loader returns documents with `page_content`
        logger.info(f"Extracted {len(text)} characters from the PDF.")  # Log extraction length
        return text
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {e}")
        logger.error(f"Failed to extract text from PDF: {e}")  # Log the error
        return ""

def split_text_into_chunks(text, chunk_size=512):
    """
    Split the extracted text into chunks of a specified size.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    logger.info(f"Split text into {len(chunks)} chunks.")  # Log the number of chunks
    return chunks

def main():
    """
    Main function to initialize and run the chatbot application.
    """

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Chatbot")
    user_question = st.text_input("Ask a question about the document:")

    # Specify the local path to the GMP PDF
    pdf_file_path = "gmpc.pdf"  # Use the correct path to the gmp.pdf file here

    if os.path.exists(pdf_file_path):
        with st.spinner("Processing GMP PDF and initializing the conversation..."):
            try:
                # Extract text from the PDF
                raw_text = extract_text_from_pdf(pdf_file_path)
                if raw_text:
                    st.success(f"Successfully extracted {len(raw_text)} characters from the PDF.")
                    # Split the extracted text into chunks based on specified size
                    text_chunks = split_text_into_chunks(raw_text)  # Split the text into chunks
                    # Generate embeddings for the text chunks
                    embeddings = generate_embeddings_for_chunks(text_chunks)
                    # Initialize Pinecone vector store and upload embeddings
                    vectorstore = initialize_pinecone_vector_store(text_chunks, embeddings)

                    if vectorstore:
                        st.success("Pinecone vector store initialized.")
                    else:
                        st.error("Failed to initialize Pinecone vector store.")

                    # Initialize conversation if not already done
                    if vectorstore and st.session_state.conversation is None:
                        st.session_state.conversation = create_conversational_chain(vectorstore)
                        st.success("Conversation chain initialized.")
                else:
                    st.error("No text extracted from the PDF.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                logger.error(f"An error occurred during processing: {e}")  # Log the error

    if user_question:
        handle_conversation(user_question)

if __name__ == '__main__':
    main()
