import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone
import requests
import tempfile
from htmlTemplates import css, bot_template, user_template

def initialize_pinecone():
    api_key = st.secrets["pinecone"]["api_key"]
    environment = st.secrets["pinecone"]["environment"]
    pinecone.init(api_key=api_key, environment=environment)
    index_name = "pdf-chat-index"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=768, metric="cosine")
    return index_name

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, index_name):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = Pinecone.from_texts(
        texts=text_chunks, 
        embedding=embeddings, 
        index_name=index_name
    )
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def fetch_pdf_from_link(url):
    response = requests.get(url)
    if response.status_code == 200:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    else:
        st.error("Failed to fetch the PDF. Please check the link.")
        return None

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with a PDF", page_icon=":book:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with a PDF :book:")
    user_question = st.text_input("Ask a question about your document:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your document")
        
        # PDF Upload Section
        pdf_doc = st.file_uploader("Upload your PDF here:", type="pdf")

        # PDF Link Section
        pdf_url = st.text_input("Or provide a link to a PDF:")

        if st.button("Process"):
            if pdf_doc is not None:
                with st.spinner("Processing uploaded PDF..."):
                    raw_text = get_pdf_text(pdf_doc)
            elif pdf_url:
                with st.spinner("Fetching and processing PDF from link..."):
                    pdf_path = fetch_pdf_from_link(pdf_url)
                    if pdf_path:
                        raw_text = get_pdf_text(pdf_path)
            else:
                st.warning("Please upload a PDF or provide a link.")
                return

            # Process the PDF content
            text_chunks = get_text_chunks(raw_text)
            index_name = initialize_pinecone()
            vectorstore = get_vectorstore(text_chunks, index_name)
            st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
