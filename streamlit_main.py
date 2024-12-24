import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import streamlit as st
import io


load_dotenv()


os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

class Chatbot:
    def __init__(self):
        # Load PDF data
        loader = PyMuPDFLoader('gpmc.pdf')
        documents = loader.load()

     
        text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=4)
        self.docs = text_splitter.split_documents(documents)

        self.embeddings = HuggingFaceEmbeddings()

 
        self.index_name = "chatbot"

     
        self.pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY'))
        # Create Pinecone index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  # Embedding dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

        
        repo_id = "google/flan-t5-xxl"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.8,
            top_k=50,
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        
        template = """
       you don't know. 

        Context: {context}
        Question: {question}
        Answer: 
        """
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Initialize Pinecone index with documents
        self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)

        # Define the retrieval-augmented generation (RAG) chain
        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question):
        return self.rag_chain.invoke(question)


# Streamlit app for querying the chatbot
def main():
    chatbot = Chatbot()

    st.title(" Chatbot")

    user_question = st.text_input("Ask a question:")

    if user_question:
        response = chatbot.ask(user_question)
        st.write(response)


if __name__ == "__main__":
    main()
import streamlit as st
import pinecone
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from os

# Access the API key from Streamlit secrets
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
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
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    
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
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Main function to set up the Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session states for conversation and chat history
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chatbot")
    user_question = st.text_input("Ask a question about document")

    if user_question:
        handle_userinput(user_question)

    # Specify the paths to your predefined PDF files
    st.sidebar.subheader("Pre-loaded Documents")

    # List the paths of predefined PDFs
    pdf_paths = [ "gmpc.pdf"]  # Update these paths as needed

    # Process the predefined PDFs
    with st.spinner("Processing pre-loaded PDFs"):
        raw_text = get_pdf_text(pdf_paths)  # Extract text from predefined PDFs

        # Split the extracted text into chunks
        text_chunks = get_text_chunks(raw_text)

        # Create the vector store based on the text chunks
        vectorstore = get_vectorstore(text_chunks)

        # Create the conversational chain
        st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
