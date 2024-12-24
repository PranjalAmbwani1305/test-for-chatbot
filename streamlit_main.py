import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pineconefrom langchain.schema import RunnablePassthrough
from langchain.schema import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pinecone import Pinecone as PineconeClient, ServerlessSpec  
from dotenv import load_dotenv
from huggingface_hub import login
from langchain.text_splitter import CharacterTextSplitter

# Log in to Hugging Face
login(token='YOUR_HUGGINGFACE_TOKEN')

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
        You are a chatbot for the Ahmedabad Government. Corporation workers will ask questions regarding the procedures in the GPMC act. 
        Answer these questions and give answers to process in a step by step process.
        If you don't know the answer, respond with: "I don't know."

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


    def ask(self, question):
        return self.rag_chain.invoke(question)                                                                                                                                   

# Streamlit UI
st.set_page_config(page_title="GPMC BOT")

# Sidebar configuration
with st.sidebar:
    st.title("Chatbot")

# Initialize session_state messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []  # Proper indentation for the block

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

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Process user input and generate response
if input_text := st.chat_input("Type your question here..."):
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.write(input_text)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response = generate_response(input_text)

            # Display the formatted response
            if isinstance(response, str) and len(response) > 100:
                st.markdown(response)
            else:
                st.write(response)

        # Append assistant's response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})
