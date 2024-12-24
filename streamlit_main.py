# Set up Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.8,
    top_k=50,
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
)  # Ensure this line is properly closed

# Chatbot class to handle queries
class Chatbot:
    def ask(self, question):
        # Retrieve relevant document chunks
        retriever = docsearch.as_retriever()
        relevant_docs = retriever.get_relevant_documents(question)  # Correct method here

        # Get the context from the relevant documents
        context = "\n".join([doc.page_content for doc in relevant_docs])  # Use 'page_content' to get the text

        # Combine context with the question to create a prompt
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

        # Generate the answer using the Hugging Face model
        response = llm(prompt=prompt)  # Pass the 'prompt' as required by HuggingFaceEndpoint
        return response

# Set up the Streamlit UI
st.title("Chatbot")

# Get user input for the query
query = st.text_input("Enter your query:")

if query:
    with st.spinner("Generating response..."):
        try:
            chatbot = Chatbot()
            response = chatbot.ask(query)
            st.write("Answer:", response)
        except Exception as e:
            st.error(f"Error: {e}")
