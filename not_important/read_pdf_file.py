from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.tools import Tool
# Load model llm
llm = ChatOllama(model="mistral")
# File loader
document = PyPDFLoader("./data/Bandar.pdf").load()
# Embedding model
embed_model = OllamaEmbeddings(model="llama3")
# Vector store index
vectorstore = FAISS.from_documents(document, embed_model)
# build chain to be ready for agent (Prepere the tool)
qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=vectorstore.as_retriever())


qa_chain_tool = Tool(
    name="ReadPDFFile",
    func=qa_chain,
    description="Use this tool to read the PDF file and answer any questions about Bandar."
)
