import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pinecone
import warnings

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone with API key and environment from .env file
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")


# Create or connect to an existing Pinecone index
index_name = "fashion-documents"  # You can choose a custom name for your index
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768)  # Set embedding dimension (768 for "all-MiniLM-L6-v2")
index = pinecone.Index(index_name)

# Load the Hugging Face model for QA
hf_hub_llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"temperature": 1, "max_new_tokens": 1024},
)

# Load the vector store (using Chroma and Hugging Face embeddings)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model)

# Define the prompt template for retrieval-based question answering
prompt_template = """
Your role is to interpret user queries and provide precise responses based on the relevant database. 
Respond clearly and concisely, without extraneous information.
"""
custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Define the RetrievalQA chain
rag_chain = RetrievalQA.from_chain_type(
    llm=hf_hub_llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(top_k=3),
    chain_type_kwargs={"prompt": custom_prompt}
)

# Function to get response from the QA system
def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    return response_text[answer_start:].strip()

# Streamlit UI
st.title("🤖 AI Assistant")

# Store LLM-generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm here to help. Ask me anything."}]

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for AI Assistant QA
if prompt := st.text_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response if last message is from user
    if st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Fetching response..."):
                response = get_response(prompt)
                st.markdown(response)

        # Append assistant's response to message history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Load and process PDF documents to create Pinecone embeddings
def process_pdf_to_pinecone(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    texts = text_splitter.split_documents(documents)

    # Convert the text chunks into embeddings
    embeddings = embedding_model.embed_documents([doc.page_content for doc in texts])

    # Prepare data for Pinecone insertion
    pinecone_data = [
        {"id": str(i), "values": embedding, "metadata": {"text": text.page_content}}
        for i, (embedding, text) in enumerate(zip(embeddings, texts))
    ]

    # Insert the embeddings into Pinecone
    index.upsert(vectors=pinecone_data)

    # Optionally, you can print a confirmation message
    print("Vector embeddings created and documents added to Pinecone index.")

# Example PDF processing (run this only once)
pdf_path = "gmpc.pdf"  # Provide your PDF path here
process_pdf_to_pinecone(pdf_path)

# Function to query Pinecone and get relevant documents
def query_pinecone(query):
    # Generate embedding for the query
    query_embedding = embedding_model.embed_query(query)

    # Perform the search
    results = index.query(queries=[query_embedding], top_k=3, include_metadata=True)

    # Print top 3 results
    print("\nTop 3 relevant results from Pinecone:")
    for result in results["results"][0]["matches"]:
        print(f"Found: {result['metadata']['text'][:200]}...")  # Print first 200 characters of each result

# Example query to test Pinecone
user_query = "What are some popular items for winter?"
query_pinecone(user_query)
