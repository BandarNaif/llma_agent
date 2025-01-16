import pandas as pd
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.tools import Tool
from langchain.schema import Document

# Load model llm
llm = ChatOllama(model="mistral")

# File loader (CSV file loading and processing)
csv_file_path = "./data/Bandar.csv"
df = pd.read_csv(csv_file_path)

# Assuming that the CSV has a column of text (for example, 'content') we want to use for embedding
# You can adjust this depending on your CSV's structure
documents = [Document(page_content=row['content']) for _, row in df.iterrows()]

# Embedding model
embed_model = OllamaEmbeddings(model="llama3")

# Vector store index
vectorstore = FAISS.from_documents(documents, embed_model)

# Build chain to be ready for agent (Prepare the tool)
qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=vectorstore.as_retriever())

qa_chain_tool = Tool(
    name="ReadCSVFile",
    func=qa_chain,
    description="Use this tool to read the CSV file and answer any questions about Bandar."
)
