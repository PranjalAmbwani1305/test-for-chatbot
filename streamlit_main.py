import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import pinecone
import huggingface_hub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

try:
    doc_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    st.success("Document store loaded from existing index!")
except Exception as e:
    st.error(f"Failed to load existing index: {e}")
    try:
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine"
        )
        st.success(f"Index '{PINECONE_INDEX_NAME}' created successfully!")
        doc_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
        st.info("Document store initialized after index creation.")
    except Exception as create_error:
        st.error(f"Failed to create index: {create_error}")
        st.stop()


st.title("Chatbot")
pdf_path = "gpmc.pdf"
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

if doc_store is not None:
    with st.spinner("Indexing document..."):
        batch_size = 32
        for i in range(0, len(text_chunks), batch_size):
            i_end = min(i + batch_size, len(text_chunks))
            batch = text_chunks[i:i_end]
            ids = [str(x) for x in range(i, i_end)]
            embeds = embeddings.embed_documents(batch)
            metadata = [{"text": text} for text in batch]
            vectors = [(ids[j], embeds[j], metadata[j]) for j in range(len(batch))]
            doc_store.upsert(vectors=vectors)
    st.success("Document indexed successfully!")

    # Prompt Template
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer:"""
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    
    llm_repo_id = "google/flan-t5-xl" 
    llm = HuggingFaceHub(repo_id=llm_repo_id, model_kwargs={"temperature":0.5, "max_length":512}, huggingfacehub_api_token=HUGGINGFACE_API_TOKEN)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=doc_store.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_PROMPT})

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
