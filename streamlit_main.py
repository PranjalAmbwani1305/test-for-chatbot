import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Initialize the Hugging Face Hub LLM
hf_hub_llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"temperature": 1, "max_new_tokens": 1024},
)

# Load the vector store from disk (Note: data directory not needed anymore)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model)

# Define the prompt template
prompt_template = """
Your role is to interpret user queries and provide precise responses based on the relevant database. 
Respond clearly and concisely, without extraneous information.
"""

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=hf_hub_llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(top_k=3),  # Retrieve top 3 results
    chain_type_kwargs={"prompt": custom_prompt}
)

def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    return answer

# Streamlit UI
st.title("🤖 AI Assistant")

# Store LLM-generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm here to help. Ask me anything."}]

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
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
