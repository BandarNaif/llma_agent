from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import Tool


# Load Documents
document_loader = PyMuPDFLoader("./data/Bandar.pdf")
document = document_loader.load()
# Embedding model
embedding = OllamaEmbeddings(model="llama3")
# Initialize a FAISS vector store
vector_store = FAISS.from_documents(document, embedding)
# Define retrieval tool using vector store
pdf_tool = Tool(
    name="Bandar Information",
    func=lambda query: vector_store.similarity_search(query, k=3),
    description=" Retrieves documents based on Bandar information"
)
