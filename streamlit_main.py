import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import pinecone
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", api_key=HUGGINGFACE_API_TOKEN)

try:
    doc_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    print("Pinecone connection successful!")
except Exception as e:
    print(f"Pinecone connection failed: {e}")
    
    try:
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine"
        )
        st.success(f"Index '{PINECONE_INDEX_NAME}' created successfully!")
        # Instantiate AFTER creating the index
        doc_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
        st.info("Document store initialized after index creation.")

    except Exception as create_error:
        st.error(f"Failed to create index: {create_error}")
        st.stop()  # Stop execution if index creation fails

st.title("Chatbot")
pdf_path = "gpmc.pdf"  # Make sure this path is correct
try:
    pdf_reader = PdfReader(pdf_path)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
except FileNotFoundError:
    st.error(f"PDF file not found: {pdf_path}")
    st.stop()

chunk_size = 500
text_chunks = [pdf_text[i : i + chunk_size] for i in range(0, len(pdf_text), chunk_size)]

if doc_store is not None: # Check if doc_store is initialized
    with st.spinner("Indexing document..."):
        batch_size = 32 # Add batching for efficiency
        for i in range(0, len(text_chunks), batch_size):
            i_end = min(i + batch_size, len(text_chunks))
            batch = text_chunks[i:i_end]
            ids = [str(x) for x in range(i, i_end)]
            embeds = embeddings.embed_documents(batch)
            metadata = [{"text": text} for text in batch]
            vectors = [(ids[j], embeds[j], metadata[j]) for j in range(len(batch))]
            doc_store.upsert(vectors=vectors)
    st.success("Document indexed successfully!")

    query = st.text_input("Enter your query:")  # Use st.text_input for user input
    if query: # Only execute if query is not empty
        try:
            results = doc_store.similarity_search(query, k=5)
            st.write(f"Search results: {results}")

            st.success("Query executed successfully!")
        except Exception as query_error:
            st.error(f"Failed to execute query: {query_error}")
else:
    st.error("Document store could not be initialized. Check previous errors.")
