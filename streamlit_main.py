import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env
load_dotenv()

# Access the API keys from Streamlit secrets
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

# Function to extract text from PDFs
def get_pdf_text(pdf_paths):
    text = ""
    for pdf in pdf_paths:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the extracted text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a Pinecone vector store using Hugging Face embeddings
def get_vectorstore(text_chunks):
    mbeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create Pinecone index for storing vectors
    index_name = "chatbot"
    
    # If the index does not exist, create it
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=embeddings.embedding_size)
    
    # Get the vector store (Pinecone index)
    index = pinecone.Index(index_name)
    
    # Insert vectors into the Pinecone index
    vectors = [embeddings.embed(text) for text in text_chunks]
    ids = [str(i) for i in range(len(text_chunks))]
    
    # Upsert the vectors into Pinecone
    index.upsert(vectors=zip(ids, vectors))
    
    # Create the vector store using the Pinecone index
    vectorstore = Pinecone(index=index, embedding_function=embeddings.embed)
    return vectorstore

# Function to create a conversational chain using Hugging Face
def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and display conversation history
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"**User**: {message.content}")
        else:
            st.write(f"**Bot**: {message.content}")

# Main function to set up the Streamlit app
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    
    # Initialize session states for conversation and chat history
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Chatbot with PDF Integration")
    user_question = st.text_input("Ask a question about the document:")

    # Process the predefined PDFs and set up conversation chain
    pdf_paths = ["gpmc.pdf"]  # Path to the pre-loaded PDF(s)

    with st.spinner("Processing pre-loaded PDFs..."):
        raw_text = get_pdf_text(pdf_paths)  # Extract text from PDFs

        # Split the extracted text into chunks
        text_chunks = get_text_chunks(raw_text)

        # Create the vector store based on the text chunks
        vectorstore = get_vectorstore(text_chunks)

        # Create the conversational chain and store it in session state
        if st.session_state.conversation is None:
            st.session_state.conversation = get_conversation_chain(vectorstore)

    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
