import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceEndpoint
from pinecone import Pinecone as PineconeClient, ServerlessSpec  
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()

# Set up API keys
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

class Chatbot:
    def __init__(self):
        # Load PDF data
        loader = PyMuPDFLoader('gpmc.pdf') 
        documents = loader.load()
        
        # Split documents into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=4)
        self.docs = text_splitter.split_documents(documents)

        # Initialize Hugging Face embeddings
        self.embeddings = HuggingFaceEmbeddings()

        # Define the index name
        self.index_name = "chatbot"

        # Initialize Pinecone client
        self.pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY')) 
        # Create Pinecone index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'  
                )
            )

        # Set up Hugging Face model
        repo_id = "hkunlp/instructor-xl"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id, 
            temperature=0.8, 
            top_k=50, 
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        # Define the updated prompt template
        template = """
        "I don't know."

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

        # Set up the Retrieval QA chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff", 
            retriever=self.docsearch.as_retriever()
        )

    def ask(self, question):
        return self.rag_chain.run(question)  # Use the correct method to query the chain

# Streamlit UI
st.set_page_config(page_title="GPMC BOT", page_icon="🤖", layout="centered")

# Add custom CSS to style the chatbot interface
st.markdown(
    """
    <style>
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        background-color: #f0f0f0;
        margin-bottom: 10px;
        font-size: 14px;
    }
    .user-message {
        background-color: #d1f7c4;
        text-align: right;
    }
    .assistant-message {
        background-color: #e1e1e1;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        color: #0077b6;
        text-align: center;
    }
    .chatbox {
        max-height: 500px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.title("GPMC BOT")
    st.markdown("Ask questions related to GPMC procedures.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Ahmedabad_GMC_Logo.svg/1024px-Ahmedabad_GMC_Logo.svg.png", use_column_width=True)

# Initialize session_state messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache the Chatbot instance
@st.cache_resource
def get_chatbot():
    return Chatbot()

# Function to clean and format the response
def generate_response(input_text):
    bot = get_chatbot()
    response = bot.ask(input_text)

    # Clean special characters and format the response
    response_parts = response.split("\n")
    formatted_response = []
    current_part = ""

    for part in response_parts:
        # Detect start of a new numbered section
        if part.strip().isdigit() or part.strip().startswith(tuple(str(i) for i in range(1, 10))) or part.strip().startswith(("404.", "405.")):
            if current_part:  # Append the previous part before starting a new one
                formatted_response.append(current_part.strip())
            current_part = f"{part.strip()} "  # Start a new numbered section
        else:
            current_part += part.strip() + " "  # Continue the current section
    
    # Append the last accumulated section if there's any leftover
    if current_part:
        formatted_response.append(current_part.strip())
    
    # Join each section with a newline and bullet point format
    return "\n\n".join(f"- {part}" for part in formatted_response)

# Display chat messages in a styled layout
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True):
            pass
    else:
        with st.markdown(f'<div class="chat-message assistant-message">{message["content"]}</div>', unsafe_allow_html=True):
            pass
st.markdown('</div>', unsafe_allow_html=True)

# Process user input and generate response
if input_text := st.chat_input("Type your question here..."):
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.markdown(f'<div class="chat-message user-message">{input_text}</div>', unsafe_allow_html=True):
        pass

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response = generate_response(input_text)

            # Display the formatted response
            if isinstance(response, str) and len(response) > 100:
                st.markdown(f'<div class="chat-message assistant-message">{response}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">{response}</div>', unsafe_allow_html=True)

        # Append assistant's response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})
