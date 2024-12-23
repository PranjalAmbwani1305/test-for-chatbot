import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone
import os

os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]


# Fixed PDF file (you can change the path to your fixed PDF)
PDF_PATH = "gpmc.pdf"  # Example: local file or URL

def get_pdf_text(pdf):
    """
    Extract text from the PDF.
    """
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Split the extracted text into smaller chunks for processing.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, index_name):
    """
    Create vector embeddings for the PDF chunks and store them in Pinecone.
    """
    embeddings = HuggingFaceInstructEmbeddings(model_name="meta-llama/Llama-3.3-70B-Instruct")
    vectorstore = Pinecone.from_texts(
        texts=text_chunks, 
        embedding=embeddings, 
        index_name=index_name
    )
    return vectorstore

def get_conversation_chain(vectorstore):
    """
    Create a conversation chain to handle the retrieval of relevant information from the vectorstore.
    """
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    """
    Process the user's question and fetch answers based on the PDF content.
    """
    if not st.session_state.conversation:
        st.error("The PDF content is not processed yet. Please reload the page.")
        return

    # Fetch response from the conversation chain
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # User's input
            st.write(f"<p style='color:blue;'>User: {message.content}</p>", unsafe_allow_html=True)
        else:
            # Bot's response
            st.write(f"<p style='color:green;'>Bot: {message.content}</p>", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat with PDF", page_icon=":book:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Load and process the fixed PDF
    with st.spinner("Loading and processing the PDF..."):
        raw_text = get_pdf_text(PDF_PATH)
        text_chunks = get_text_chunks(raw_text)
        index_name = "pdf_index"  # You can change the index name
        vectorstore = get_vectorstore(text_chunks, index_name)
        st.session_state.conversation = get_conversation_chain(vectorstore)

    # Ask the user for a question
    st.header("Chat with your PDF :book:")
    user_question = st.text_input("Ask a question about the document:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
