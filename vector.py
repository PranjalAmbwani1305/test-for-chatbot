import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone

# Load environment variables from .env file
load_dotenv()

# Load Pinecone API key and environment from .env
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")  
# Initialize Pinecone connection
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# Create or connect to an existing Pinecone index
index_name = "fashion-documents"  # You can choose a custom name for your index
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768)  # Set the embedding dimension (768 for "all-MiniLM-L6-v2")

index = pinecone.Index(index_name)

# Load the PDF
pdf_path = "gmpc.pdf"  # Provide your PDF path here
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
texts = text_splitter.split_documents(documents)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

# Example query to test the vector store
user_query = "What are some popular items for winter?"
query_pinecone(user_query)
