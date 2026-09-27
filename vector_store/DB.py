from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(page_content="Practical considerations: best practices for model training, hyperparameter tuning, and GPU acceleration.", metadata={"source": "Fundamentals of Deep Learning", "section": "Practical Considerations"}),
    Document(page_content="Case studies and examples: hands-on examples and TensorFlow code snippets demonstrating various deep learning models.", metadata={"source": "Fundamentals of Deep Learning", "section": "Case Studies and Examples"}),
    Document(page_content="Appendices: Preface explains objectives and prerequisites; Table of Contents lists chapters; Index provides key terms; Conclusion highlights practical deep learning with TensorFlow.", metadata={"source": "Fundamentals of Deep Learning", "section": "Appendices and Supplementary Material"})
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = Chroma(
    collection_name='deep_learning',
    embedding_function=embeddings,
    persist_directory="chroma_langchain_db",
)

vector_store.add_documents(documents=docs)

result = vector_store.similarity_search('best practices for model training', k=2)

for res in result:
    print(res.page_content)
    print(res.metadata)

retriever = vector_store.as_retriever()



