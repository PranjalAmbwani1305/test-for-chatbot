def main():
    """
    Main function to initialize and run the chatbot application.
    """
    st.set_page_config(page_title="Chat with GMP PDF", page_icon="📚")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Chatbot with GMP PDF Document Integration")
    user_question = st.text_input("Ask a question about the document:")

    # Assuming the file is stored locally at the specified path
    pdf_file_path = "gmpc.pdf"  # Specify the local PDF file path

    if os.path.exists(pdf_file_path):
        with st.spinner("Processing GMP PDF and initializing the conversation..."):
            try:
                raw_text = extract_text_from_pdf(pdf_file_path)
                if raw_text:
                    text_chunks = raw_text.split("\n")  # Split into chunks based on newlines
                    embeddings = generate_embeddings_for_chunks(text_chunks)
                    vectorstore = initialize_pinecone_vector_store(text_chunks, embeddings)

                    if vectorstore and st.session_state.conversation is None:
                        st.session_state.conversation = create_conversational_chain(vectorstore)
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

    if user_question:
        handle_conversation(user_question)
