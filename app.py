import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # Updated import
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

# Always use explicit model name to prevent dimension mismatch
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "DOCS"
DB_DIR = "chroma_db"

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    # Use delete=False and cleanup explicitly to avoid file access errors on Windows
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):
        with st.spinner("Processing document..."):
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Filter out any empty page contents that might pass None to HuggingFace
            valid_docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(valid_docs)

            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=DB_DIR
            )
            # Automatic persistence in langchain-chroma (no vectorstore.persist() needed)

        st.success("Vector database created!")
        
        # Cleanup temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

if os.path.exists(DB_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(model="mistral-small-2506")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    if query:
        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt)

        st.write("### AI Answer")
        st.write(response.content)