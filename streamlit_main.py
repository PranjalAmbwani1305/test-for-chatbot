import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from huggingface_hub import login

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HUGGINGFACE_REPO_ID = os.getenv("HUGGINGFACE_REPO_ID")

login(HUGGINGFACE_API_TOKEN)



pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY')) 
        # Create Pinecone index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=self.index_name,
        dimension=346,  
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'  
                )
            )

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error initializing embeddings: {e}")
    st.stop()

# Set up the chatbot interface
st.title("Chatbot")
pdf_path = "gpmc.pdf"

# Load and read the PDF document
try:
    pdf_reader = PdfReader(pdf_path)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
except FileNotFoundError:
    st.error(f"PDF file not found: {pdf_path}")
    st.stop()
except Exception as e:
    st.error(f"Error reading PDF: {e}")
    st.stop()

# Chunk the text into smaller parts for embedding
chunk_size = 500
text_chunks = [pdf_text[i:i + chunk_size] for i in range(0, len(pdf_text), chunk_size)]

# Index the document text into Pinecone
if doc_store is not None:
    with st.spinner("Indexing document..."):
        batch_size = 32
        for i in range(0, len(text_chunks), batch_size):
            i_end = min(i + batch_size, len(text_chunks))
            batch = text_chunks[i:i_end]
            ids = [str(i + j) for j in range(len(batch))]
            embeds = embeddings.embed_documents(batch)
            metadata = [{"text": text} for text in batch]
            vectors = [(ids[j], embeds[j], metadata[j]) for j in range(len(batch))]
            
            try:
                index.upsert(vectors=vectors)
            except Exception as e:
                st.error(f"Error during upsert: {e}")
                st.stop()
        st.success("Document indexed successfully!")

    # Set up the question-answering system
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer:"""
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    # Initialize the language model for QA
    llm = HuggingFaceHub(repo_id=HUGGINGFACE_REPO_ID, model_kwargs={"temperature": 0.5, "max_length": 1024}, huggingfacehub_api_token=HUGGINGFACE_API_TOKEN)

    # Set up the QA chain with the retriever from Pinecone
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=doc_store.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_PROMPT})

    # Get user input for the query
    query = st.text_input("Enter your query:")
    if query:
        with st.spinner("Generating response..."):
            try:
                result = qa({"query": query})
                st.write("Answer:", result["result"])
                with st.expander("Source Documents"):
                    st.write(result["source_documents"])
                st.success("Query executed successfully!")
            except Exception as query_error:
                st.error(f"Failed to execute query: {query_error}")
else:
    st.error("Document store could not be initialized. Check previous errors.")
