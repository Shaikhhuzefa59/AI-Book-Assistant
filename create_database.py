from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# Load the PDF
loader = PyPDFLoader('document loaders/fundamentals-of-deep-learning.pdf')
docs = loader.load()

# Filter empty documents
valid_docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(valid_docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

# Store into ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name='DOCS',
    persist_directory='chroma_db'
)