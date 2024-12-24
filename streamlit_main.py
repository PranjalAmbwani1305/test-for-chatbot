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

# Load environment variables
load_dotenv()

def generate_embeddings_for_chunks(text_chunks):
    """
    Generate embeddings for each chunk of text using HuggingFace model.
    """
    embeddings_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    return [embeddings_model.embed(chunk) for chunk in text_chunks]

def initialize_pinecone_vector_store(text_chunks, embeddings):
    """
    Initialize the Pinecone vector store and upload embeddings.
    """
    index_name = "chatbot"
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENV")

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    pinecone_client = pinecone

    if index_name not in pinecone_client.list_indexes():
        pinecone_client.create_index(index_name, dimension=len(embeddings[0]))

    pinecone_index = pinecone_client.Index(index_name)
    ids = [str(i) for i in range(len(text_chunks))]
    pinecone_index.upsert(vectors=zip(ids, embeddings))

    return Pinecone(index=pinecone_index, embedding_function=generate_embeddings_for_chunks)

def create_conversational_chain(vectorstore):
    """
    Create a conversational chain using HuggingFace LLM and vector store retriever.
    """
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def handle_conversation(user_question):
    """
    Handle user queries using the conversational chain.
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"**User**: {message.content}")
        else:
            st.write(f"**Bot**: {message.content}")

def extract_text_from_pdf(pdf_file_path):
    """
    Extract text from the given PDF file using PyMuPDF.
    """
    loader = PyMuPDFLoader(pdf_file_path)
    return loader.load()

def main():
    """
    Main function to initialize and run the chatbot application.
    """
    st.set_page_config(page_title="Chat with GMP PDF", page_icon="📚")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Chatbot with GMP PDF Document Integration")
    user_question = st.text_input("Ask a question about the document:")

    # Specify the local PDF file path
    pdf_file_path = "gmpc.pdf" 

    if os.path.exists(pdf_file_path):
        with st.spinner("Processing GMP PDF and initializing the conversation..."):
            raw_text = extract_text_from_pdf(pdf_file_path)
            text_chunks = raw_text.split("\n")  # Split into chunks based on newlines
            embeddings = generate_embeddings_for_chunks(text_chunks)
            vectorstore = initialize_pinecone_vector_store(text_chunks, embeddings)
            
            if vectorstore and st.session_state.conversation is None:
                st.session_state.conversation = create_conversational_chain(vectorstore)

    if user_question:
        handle_conversation(user_question)

if __name__ == '__main__':
    main()
